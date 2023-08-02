#!\usr\AppData\Local\Programs\Python python3.11

from flask import Flask, render_template, request, flash, Response
import os
import dotenv
import cv2
import numpy as np
from werkzeug.utils import redirect, secure_filename
from my_model import init_tracker, DetectionPredictor


app = Flask(__name__)
dotenv.load_dotenv("env_vars.env")

UPLOAD_FOLDER = "uploads"
ALLOWED_EXTENSIONS ={"mp4","avi","mkv"}

app.secret_key = os.getenv("SECRET_KEY")
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024
app.config['UPLOAD_FOLDER']=UPLOAD_FOLDER

predictor = DetectionPredictor()
init_tracker()

new_file = None
class UploadedFile:
    def __init__(self, file_name=None, text=None):
        self.is_playing = True
        self.file_name = file_name
        self.text = text
        
    def generate_frames(self):
        while True:
            if self.is_playing:
                video = predictor("uploads/" + self.file_name)
                for frame in video:
                    ret, buffer = cv2.imencode('.jpg', np.array(frame[0]))
                    frame = buffer.tobytes()

                    yield (b'--frame\r\n'
                        b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    
    def generate_frames_from_text(self):
        while True:
            if self.is_playing:
                video = predictor(self.text)
                for frame in video:
                    ret, buffer = cv2.imencode('.jpg', np.array(frame[0]))
                    frame = buffer.tobytes()

                    yield (b'--frame\r\n'
                        b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
    

    def play(self):
        self.is_playing = True

    def stop(self):
        self.is_playing = False


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route("/" )
def home():
    return render_template("index.html")#, data=[{'model': 'yolov8n'}, {'model': 'yolov8s'}, {'model': 'yolov8m'}, {'model': 'yolov8l'}, {'model': 'yolov8x'}, {'model': 'yolov8x6'}])

@app.route("/submit", methods = ["POST"])
def submit():
    global new_file
    if request.method == "POST":
        uploaded_video = request.files.get("video")
        text = request.form.get("text")
        
        if uploaded_video and not text:
            file_name = secure_filename(uploaded_video.filename)
            
            if allowed_file(file_name):
                flash('Processing submitted video')
                new_file = UploadedFile(file_name=file_name)
                uploaded_video.save(f"uploads/{new_file.file_name}")
                return render_template("submission.html")
            else:
                flash("Incorrect File Type")
                
        elif not uploaded_video and text:
            new_file = UploadedFile(text=text)
            flash('Processing submitted stream')
            return render_template("submission2.html")
        
        else:
            flash("Either upload a video, or insert a stream in the text box. Please don't do both.")

@app.route('/video_feed')
def video_feed():
    global new_file
    return Response(new_file.generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/video_feed2')
def video_feed2():
    global new_file
    return Response(new_file.generate_frames_from_text(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/play')
def play():
    global new_file
    new_file.play()
    return Response('Playback started')

@app.route('/pause')
def pause():
    global new_file
    new_file.stop()
    return Response('Playback stopped')



if __name__ == "__main__":
    app.run(host='localhost',port=8080)
