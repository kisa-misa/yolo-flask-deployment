# Ultralytics YOLO 🚀, GPL-3.0 license
"""
Run prediction on images, videos, directories, globs, YouTube, webcam, streams, etc.
Usage - sources:
    $ yolo task=... mode=predict  model=s.pt --source 0                         # webcam
                                                img.jpg                         # image
                                                vid.mp4                         # video
                                                screen                          # screenshot
                                                path/                           # directory
                                                list.txt                        # list of images
                                                list.streams                    # list of streams
                                                'path/*.jpg'                    # glob
                                                'https://youtu.be/Zgi9g1ksQHc'  # YouTube
                                                'rtsp://example.com/media.mp4'  # RTSP, RTMP, HTTP stream
Usage - formats:
    $ yolo task=... mode=predict --weights yolov8n.pt          # PyTorch
                                    yolov8n.torchscript        # TorchScript
                                    yolov8n.onnx               # ONNX Runtime or OpenCV DNN with --dnn
                                    yolov8n_openvino_model     # OpenVINO
                                    yolov8n.engine             # TensorRT
                                    yolov8n.mlmodel            # CoreML (macOS-only)
                                    yolov8n_saved_model        # TensorFlow SavedModel
                                    yolov8n.pb                 # TensorFlow GraphDef
                                    yolov8n.tflite             # TensorFlow Lite
                                    yolov8n_edgetpu.tflite     # TensorFlow Edge TPU
                                    yolov8n_paddle_model       # PaddlePaddle
    """
import platform
from collections import defaultdict
from pathlib import Path

import cv2
import torch
import numpy as np

from ultralytics.nn.autobackend import AutoBackend
from ultralytics.yolo.configs import get_config
from ultralytics.yolo.data.dataloaders.stream_loaders import LoadImages
from ultralytics.yolo.data.utils import IMG_FORMATS, VID_FORMATS
from ultralytics.yolo.utils import LOGGER, SETTINGS, colorstr, ops
from ultralytics.yolo.utils.checks import check_file, check_imgsz, check_imshow
from ultralytics.yolo.utils.files import increment_path
from ultralytics.yolo.utils.torch_utils import select_device, smart_inference_mode


class BasePredictor:

    def __init__(self, filename):

        self.project = None
        self.task = "detect"
        self.name = None
        self.exist_ok = False
        project = self.project or Path(SETTINGS['runs_dir']) / self.task
        name = self.name or f"predict"
        self.save_dir = increment_path(Path(project) / name, exist_ok=self.exist_ok)

        # Usable if setup is done
        self.model = None
        self.data = None #тут другое было
        self.device = None
        self.dataset = None
        self.vid_path, self.vid_writer = None, None
        self.source= 'uploads/'+filename # source directory for images or videos
        self.imgsz = None
        self.done_setup = False
        self.seen = None
        self.windows = None
        self.dt = None
        self.annotator = None
        self.all_outputs = None
        self.half = False
        self.save_txt = False # save results as .txt file
        self.save_conf = False # save results with confidence scores
        self.save_crop = False # save cropped images with results

    


    def preprocess(self, img):
        pass

    def get_annotator(self, img):
        raise NotImplementedError("get_annotator function needs to be implemented")

    def write_results(self, pred, batch, print_string):
        raise NotImplementedError("print_results function needs to be implemented")

    def postprocess(self, preds, img, orig_img):
        return preds

    def setup(self, source=None, model=None):
        # source
        source = str(source if source is not None else self.source)
        is_file = Path(source).suffix[1:] in (IMG_FORMATS + VID_FORMATS)
        is_url = source.lower().startswith(('rtsp://', 'rtmp://', 'http://', 'https://'))
        if is_url and is_file:
            source = check_file(source)  # download

        # model
        device = torch.device("cuda:0")# if torch.cuda.is_available() else "cpu")
        #device = select_device(self.device)
        print(device.type)
        self.half &= device.type != 'cpu'  # half precision only supported on CUDA

        model = "yolov8x6.pt"
        model = AutoBackend(model, device=device, dnn=False, fp16=False)
        stride, pt = model.stride, model.pt
        imgsz = check_imgsz(640, stride=stride)  # check image size

        # Dataloader
        bs = 1  # batch_size
        self.dataset = LoadImages(source,
                                    imgsz=imgsz,
                                    stride=stride,
                                    auto=pt,
                                    transforms=getattr(model.model, 'transforms', None),
                                    vid_stride=1)
        self.vid_path, self.vid_writer = [None] * bs, [None] * bs
        model.warmup(imgsz=(1 if pt or model.triton else bs, 3, 640, 640))  # warmup

        self.model = model
        self.imgsz = imgsz
        self.done_setup = True
        self.device = device

        return model

    @smart_inference_mode()
    def __call__(self, source=None, model=None):
        model = self.model if self.done_setup else self.setup(source, model)
        model.eval()
        self.seen, self.windows, self.dt = 0, [], (ops.Profile(), ops.Profile(), ops.Profile())
        self.all_outputs = []
        for batch in self.dataset:
            path, im, im0s, vid_cap, s = batch
            with self.dt[0]:
                im = self.preprocess(im)
                if len(im.shape) == 3:
                    im = im[None]  # expand for batch dim

            # Inference
            with self.dt[1]:
                preds = model(im, augment=False, visualize=False)

            # postprocess
            with self.dt[2]:
                preds = self.postprocess(preds, im, im0s)

            
            for i in range(len(im)):
                p = Path(path)

                log_string, log_string2, pulse_df, vehicle_df = self.write_results(i, preds, (p, im, im0s))
                s += log_string
                if len(pulse_df)>0:
                    pulse = pulse_df['pulse'][len(pulse_df)-1]
                else: pulse = 0
                self.save_preds(vid_cap, i, 'static/videos/result.mp4', vehicle_df, log_string2, pulse)#str(self.save_dir / p.name))

            
            # Print time (inference-only)
            LOGGER.info(f"{s}{'' if len(preds) else '(no detections), '}{self.dt[1].dt * 1E3:.1f}ms")

        return self.all_outputs, pulse_df

    def show(self, p):
        im0 = self.annotator.result()
        cv2.imshow(str(p), im0)
        cv2.waitKey(1)  # 1 millisecond

    def save_preds(self, vid_cap, idx, save_path, vehicle_df, log_string2, pulse):
        im0 = self.annotator.result()
        # save imgs
        if self.dataset.mode == 'image':
            cv2.imwrite(save_path, im0)
        else:  # 'video' or 'stream'
            if self.vid_path[idx] != save_path:  # new video
                self.vid_path[idx] = save_path
                if isinstance(self.vid_writer[idx], cv2.VideoWriter):
                    self.vid_writer[idx].release()  # release previous video writer
                if vid_cap:  # video
                    fps = vid_cap.get(cv2.CAP_PROP_FPS)
                    w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                else:  # stream
                    fps, w, h = 30, im0.shape[1], im0.shape[0]
                save_path = str(Path(save_path).with_suffix('.mp4'))  # force *.mp4 suffix on results videos

                self.vid_writer[idx] = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'avc1'), fps, (1920, 1080))
            height, width = im0.shape[:2]
            image_toresize = cv2.resize(im0, dsize=(1536,864)) 
            height, width = image_toresize.shape[:2]
            blank_image = np.zeros((1080,1920,3), np.uint8)#360 640
            blank_image[:,:] = (255,255,255)

            l_img = blank_image.copy()          #(height,width+200,3)
    
            # Here, y_offset+height <= blank_image.shape[0] and x_offset+width <= blank_image.shape[1]
            l_img[0:height, 0:width] = image_toresize.copy()
            thickness = 2
            count = 1
            cols = vehicle_df.columns
            cv2.putText(l_img, f"{cols[0]} {cols[1]} {cols[2]} {cols[3]}", (width+20, count*50), 0, 1, [0, 0, 0], thickness, lineType=cv2.LINE_AA)
           
            for item in vehicle_df[cols[0]]:
                count += 1
                cv2.putText(l_img, f'{item}', (width + 20, count*50), 0, 1, [0, 0, 0], thickness, lineType=cv2.LINE_AA)

            count = 1
            for item in vehicle_df[cols[1]]:
                count += 1
                cv2.putText(l_img, f'{item}', (width + 75, count*50), 0, 1, [0, 0, 0], thickness, lineType=cv2.LINE_AA)

            count = 1
            for item in vehicle_df[cols[2]]:
                count += 1
                cv2.putText(l_img, f'{item}', (width + 180, count*50), 0, 1, [0, 0, 0], thickness, lineType=cv2.LINE_AA)

            count = 1
            for item in vehicle_df[cols[3]]:
                count += 1
                cv2.putText(l_img, f'{item}', (width + 300, count*50), 0, 1, [0, 0, 0], thickness, lineType=cv2.LINE_AA)

            
            cv2.putText(l_img, f'Pulse: {pulse}', (20, height+50), 0, 1, [0, 0, 0], thickness, lineType=cv2.LINE_AA)
            cv2.putText(l_img, log_string2, (20, height+100), 0, 1, [0, 0, 0], thickness, lineType=cv2.LINE_AA)
            self.vid_writer[idx].write(l_img)


