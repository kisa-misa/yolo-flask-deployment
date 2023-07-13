#!\usr\AppData\Local\Programs\Python python3.11

from flask import Flask,render_template,request,flash,redirect, url_for, send_file, Response
#from camera import VideoCamera
import os
import my_model
import cv2
from werkzeug.utils import redirect, secure_filename
filename = ""
import pandas as pd
#import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns 
import matplotlib
import torch
import io
import base64
from pathlib import Path
from my_model import init_tracker, DetectionPredictor
from ultralytics.nn.autobackend import AutoBackend
from ultralytics.yolo.data.dataloaders.stream_loaders import LoadImages, LoadStreams
from ultralytics.yolo.data.utils import IMG_FORMATS, VID_FORMATS
from ultralytics.yolo.utils.checks import check_file, check_imgsz

matplotlib.use('agg')



# Create flask app
app = Flask(__name__)
# for encrypting the session
app.secret_key = "secret key" 
#setting Uploads Folder
UPLOAD_FOLDER = os.path.join(app.root_path, 'uploads')

app.config['UPLOAD_FOLDER']=UPLOAD_FOLDER
#Setting allowed extensions (i.e. to allow for videos only)
ALLOWED_EXTENSIONS ={"mp4","avi","mkv"} 
#Uploads of up to 10Mbs only
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024
video = []
#video_stream = VideoCamera()

is_playing = True
predictor = DetectionPredictor()

def predict(filename):
    #delete_files()
    print("Predicting...")
    init_tracker()

    #source =  + filename
    #source = str(source if source is not None else self.source)

    return predictor(filename)

def generate_frames(video):
    global is_playing
    while True:
        for frame in video:
            if is_playing:

                ret, buffer = cv2.imencode('.jpg', frame)
                frame = buffer.tobytes()

                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

        is_playing = False





def allowed_file(filename):
    return '.' in filename and \
        filename.rsplit('.',1)[1].lower() in ALLOWED_EXTENSIONS

@app.route("/" )
def hello():
    return render_template("index.html")#, data=[{'model': 'yolov8n'}, {'model': 'yolov8s'}, {'model': 'yolov8m'}, {'model': 'yolov8l'}, {'model': 'yolov8x'}, {'model': 'yolov8x6'}])


 #route to accept the video file 
@app.route("/submit", methods = ["POST","GET"])
def submit():
    global filename
    global video

    #model_name = request.form.get('comp_select')

    if request.method=="POST":
        #Checking if request has a file part
        # if 'video' not in request.files:
        #     flash("No file part")
        #     return redirect(request.url)
        #Getting submitted file    
        #file = request.files['video']
        text = request.form['text']
        #Checking if user submitted a file (The browser submits an empty file with no filename if the user doesn't select a file)
        # if file.filename =="":
        #     flash("No selected file")
        #     return redirect(request.url)
        #
        #checking if file has the allowed extension
        #if file and allowed_file(file.filename):
            #cleaning up the filename and making sure if doesn't cause any dangerous operations in the server's file directory
            #filename = secure_filename (file.filename)
            #saving file to uploads folder
            #file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))

            #loading video into model
        flash('Processing submitted video',"submission")
        outputs, pulse_df, video = predict(text)

        pulse_df.index = pd.to_datetime(pulse_df['time'], format = '%H:%M:%S  %d.%m.%Y', errors = 'coerce')

        fig, ax = plt.subplots()
        sns.lineplot(pulse_df)
        ax.set_xticklabels([pd.to_datetime(t.get_text(), errors = 'coerce').strftime('%H:%M:%S') for t in ax.get_xticklabels()])
        plt.ylabel('Pulse')
        plt.xlabel('time')
        buffer = io.BytesIO()
        plt.savefig(buffer, format="png")  # Corrected line
        buffer.seek(0)
        image_memory = base64.b64encode(buffer.getvalue())
        return  render_template("submission.html",n={"filename":filename}, img_data=image_memory.decode('utf-8'))
        
        #else if file type not allowed
        # flash("File Type Not Allowed. \".mp4\",\".avi\",\".mkv\" only")
        # return redirect(request.url)
     #TODO If user goes straight to /submit
    return render_template("submission.html")

@app.route('/video_feed')
def video_feed():
    global video
    return Response(generate_frames(video), mimetype='multipart/x-mixed-replace; boundary=frame')


# @app.route('/video')
# def stream_video():
#     return send_file('static/videos/result.mp4', mimetype='video/mp4')

@app.route('/play')
def play():
    global is_playing
    is_playing = True
    return Response('Playback started')

@app.route('/pause')
def pause():
    global is_playing
    is_playing = not is_playing
    return Response('Playback paused')



if __name__ == "__main__":
    app.run(host='127.0.0.1',port=5000)
