import os
HOME = os.getcwd()
print(HOME)

from IPython import display
display.clear_output()

import ultralytics
ultralytics.checks()

from ultralytics import YOLO

from IPython.display import display, Image

!mkdir {HOME}/datasets
%cd {HOME}/datasets

!pip install roboflow --quiet

from roboflow import Roboflow
rf = Roboflow(api_key="XPtk6Tdww8uqs7Lhr4h2")
project = rf.workspace("exjobb-dq06p").project("vehicles-k83q3")
dataset = project.version(2).download("yolov8")


%cd {HOME}

!yolo task=detect mode=train model=yolov8x6.pt data={dataset.location}/data.yaml epochs=2 imgsz=800 plots=True