#!\usr\AppData\Local\Programs\Python python3.11

from flask import Flask,render_template,request,flash,redirect, Response
import os
import cv2
from werkzeug.utils import redirect, secure_filename
filename = ""
from my_model import init_tracker, DetectionPredictor


app = Flask(__name__)
app.secret_key = "secret key"
UPLOAD_FOLDER = os.path.join(app.root_path, 'uploads')

app.config['UPLOAD_FOLDER']=UPLOAD_FOLDER
ALLOWED_EXTENSIONS ={"mp4","avi","mkv"}
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024
video = []

is_playing = True
predictor = DetectionPredictor()
init_tracker()

def generate_frames():
    global is_playing
    _, video = predictor('./uploads/' + filename)
    while True:
        if is_playing:
            for frame in video:

                ret, buffer = cv2.imencode('.jpg', frame)
                frame = buffer.tobytes()

                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

            is_playing = False


def generate_frames2():
    global is_playing
    while True:
        if is_playing:
            _, video = predictor(text)
            for frame in video:
                cv2.imshow('frame', frame)
                print(123)
                ret, buffer = cv2.imencode('.jpg', frame)
                frame = buffer.tobytes()

                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')



def allowed_file(filename):
    return '.' in filename and \
        filename.rsplit('.',1)[1].lower() in ALLOWED_EXTENSIONS

@app.route("/" )
def hello():
    return render_template("index.html")#, data=[{'model': 'yolov8n'}, {'model': 'yolov8s'}, {'model': 'yolov8m'}, {'model': 'yolov8l'}, {'model': 'yolov8x'}, {'model': 'yolov8x6'}])

@app.route("/submit", methods = ["POST","GET"])
def submit():
    global filename
    global text
    global is_playing
    is_playing = True
  # model_name = request.form.get('comp_select')

    if request.method == "POST":
        if 'video' not in request.files and 'text' not in request.form:
            flash("No file part")
            return redirect(request.url)
        file = request.files['video']
        text = request.form['text']
        if file.filename =="" and text=="":
            flash("No selected file")
            return redirect(request.url)

        elif file.filename !="":
            if file and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))

            # loading video into model
            flash('Processing submitted video', "submission")
            return render_template("submission.html")
        else:
            flash('Processing submitted stream', "submission2")
            return render_template("submission2.html")


        # else if file type not allowed
        # flash("File Type Not Allowed. \".mp4\",\".avi\",\".mkv\" only")
        # return redirect(request.url)
    # TODO If user goes straight to /submit
    return render_template("submission2.html")

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/video_feed2')
def video_feed2():
    return Response(generate_frames2(), mimetype='multipart/x-mixed-replace; boundary=frame')

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
