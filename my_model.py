from ultralytics.yolo.v8.detect.predict import estimatespeed, init_tracker, xyxy_to_xywh, compute_color_for_labels, draw_border, UI_box, ccw #, draw_boxes
from ultralytics.yolo.engine.predictor import BasePredictor
from ultralytics.yolo.utils.plotting import Annotator
from ultralytics.yolo.utils import ops
from collections import deque
import pandas as pd
import os
import time
import torch
import numpy as np

from ultralytics.yolo.v8.detect.deep_sort_pytorch.utils.parser import get_config
from ultralytics.yolo.v8.detect.deep_sort_pytorch.deep_sort import DeepSort
DEFAULT_CONFIG = "ultralytics/yolo/configs/default.yaml"

palette = (2 ** 11 - 1, 2 ** 15 - 1, 2 ** 20 - 1)
deq = {}
indices = [0] * 1000
c = 0
num = 1
deepsort = None
object_counter = {}
speed_line_queue = {}
vehicle_df = pd.DataFrame(columns=["id", "class", "speed", "weight"])
pulse_df = pd.DataFrame(columns=["time", "pulse"])

def init_tracker():
    global deepsort
    cfg_deep = get_config()
    cfg_deep.merge_from_file("ultralytics/yolo/v8/detect/deep_sort_pytorch/configs/deep_sort.yaml")

    deepsort= DeepSort(cfg_deep.DEEPSORT.REID_CKPT,
                            max_dist=cfg_deep.DEEPSORT.MAX_DIST, min_confidence=cfg_deep.DEEPSORT.MIN_CONFIDENCE,
                            nms_max_overlap=cfg_deep.DEEPSORT.NMS_MAX_OVERLAP, max_iou_distance=cfg_deep.DEEPSORT.MAX_IOU_DISTANCE,
                            max_age=cfg_deep.DEEPSORT.MAX_AGE, n_init=cfg_deep.DEEPSORT.N_INIT, nn_budget=cfg_deep.DEEPSORT.NN_BUDGET,
                            use_cuda=True)

def draw_boxes(img, bbox, names,object_id, identities=None, offset=(0, 0)):
    height, width, _ = img.shape
    # remove tracked point from buffer if object is lost
    global c
    global pulse_df
    
    for key in list(deq):
        if key not in identities:
            deq.pop(key)

    weights = [0,0,7,2,0,30,0,19]
    speeds = [0] * 8
    vehicle_constants = [0,0,2,1,0,6,0,4]
    vehicle_df = pd.DataFrame(columns=["id", "class", "speed", "weight"])


    for i, box in enumerate(bbox):
        obj_name = "bike" if object_id[i]==3 else names[object_id[i]]
        x1, y1, x2, y2 = [int(i) for i in box]
        x1 += offset[0]
        x2 += offset[0]
        y1 += offset[1]
        y2 += offset[1]

        # code to find center of bottom edge
        center = (int((x2+x1)/ 2), int((y2+y2)/2))

        # get ID of object

        id = int(identities[i]) if identities is not None else 0

        # create new buffer for new object
        if id not in deq:  
            deq[id] = deque(maxlen= 64)
            if object_id[i] in [2, 3, 5, 7]:
              c +=1
              indices[id] = c
            speed_line_queue[id] = []
        color = compute_color_for_labels(object_id[i])
        
        
        label = '{}{:d}'.format("", indices[id]) + ":"+ '%s' % (obj_name)
        

        # add center to buffer
        deq[id].appendleft(center)
        if len(deq[id]) >= 2:
            object_speed = estimatespeed(deq[id][1], deq[id][0], x2-x1, y2-y1)
            speed_line_queue[id].append(object_speed)
            if obj_name not in object_counter:
                    object_counter[obj_name] = 1
        
        #motorcycle_weight = 1.638
        #car_weight = 6.72
        #truck_weight = 18.75
        #bus_weight = 30

        try:
            spd = sum(speed_line_queue[id])//len(speed_line_queue[id])*vehicle_constants[object_id[i]]
            weight = weights[object_id[i]]
            row = [indices[id], obj_name, spd, weight]
            vehicle_df.loc[len(vehicle_df)] = row
            speeds[object_id[i]] += spd
            label = label + " v=" + str(spd) + " m=" + str(weight)

        except:
            pass
        UI_box(box, img, label=label, color=color, line_thickness=2)

    t = time.localtime()
    current_time = time.strftime("%H:%M:%S %d.%m.%Y", t)
    pulse = sum(np.multiply(speeds, weights))
    row = [current_time, pulse]
    pulse_df.loc[len(pulse_df)] = row

    return img, pulse_df, vehicle_df



class DetectionPredictor(BasePredictor):

    def get_annotator(self, img):
        return Annotator(img, line_width=3, example=str(self.model.names))

    def preprocess(self, img):
        img = torch.from_numpy(img).to(self.model.device)
        img = img.half() if self.model.fp16 else img.float()  # uint8 to fp16/32
        img /= 255  # 0 - 255 to 0.0 - 1.0
        return img

    def postprocess(self, preds, img, orig_img):
        preds = ops.non_max_suppression(preds,
                                        0.25,
                                        0.7,
                                        classes = [2, 3, 5, 7],
                                        agnostic=False,
                                        max_det=300)

        for i, pred in enumerate(preds):
            shape = orig_img[i].shape if self.webcam else orig_img.shape
            pred[:, :4] = ops.scale_boxes(img.shape[2:], pred[:, :4], shape).round()

        return preds

    def write_results(self, idx, preds, batch):
        bbox_xyxy = []
        identities = []
        object_id = []
        global vehicle_df
        global pulse_df
        global num
        p, im, im0 = batch
        all_outputs = []
        log_string = ""
        log_string2 = ""
        if len(im.shape) == 3:
            im = im[None]  # expand for batch dim
        self.seen += 1
        im0 = im0.copy()
        
        frame = getattr(self.dataset, 'frame', 0)

        self.data_path = p
        #save_path = str(self.save_dir / p.name)  # im.jpg
        #self.txt_path = str(self.save_dir / 'labels' / p.stem) + ('' if self.dataset.mode == 'image' else f'_{frame}')
        log_string += '%gx%g ' % im.shape[2:]  # print string
        self.annotator = self.get_annotator(im0)

        det = preds[idx]
        all_outputs.append(det)
        if len(det) == 0:
            return log_string, log_string2, pulse_df, vehicle_df,
        
        #count = 0
        for c in det[:, 5].unique():
            #count += 1
            n = (det[:, 5] == c).sum()  # detections per class
            cl = self.model.names[int(c)]
            if cl=="motorcycle": cl = "bike"
            log_string2 += f"{n} {cl}{'s' * (n > 1)}, "
        # write
        gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
        xywh_bboxs = []
        confs = []
        oids = []
        outputs = []
        for *xyxy, conf, cls in reversed(det):
            x_c, y_c, bbox_w, bbox_h = xyxy_to_xywh(*xyxy)
            xywh_obj = [x_c, y_c, bbox_w, bbox_h]
            xywh_bboxs.append(xywh_obj)
            confs.append([conf.item()])
            oids.append(int(cls))
        xywhs = torch.Tensor(xywh_bboxs)
        confss = torch.Tensor(confs)
          
        outputs = deepsort.update(xywhs, confss, oids, im0)
        

        if len(outputs) > 0:
            bbox_xyxy = outputs[:, :4]
            identities = outputs[:, -2]
            object_id = outputs[:, -1]

        img, pulse_df, vehicle_df = draw_boxes(im0, bbox_xyxy, self.model.names, object_id, identities) 
      
        return log_string + log_string2, log_string2, pulse_df, vehicle_df


#deleting frames from a previous analysis
def delete_files():
  for file in os.listdir('static'):
    full_path='static/'+file
    if os.path.exists(full_path):
      os.remove(full_path)
    else:
      print("The file does not exist")  

