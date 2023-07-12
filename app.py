from flask import Flask,render_template,request,flash,redirect, url_for, send_file
import os
import my_model
from werkzeug.utils import redirect, secure_filename
filename = ""
import pandas as pd
#import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns 
import matplotlib
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

def allowed_file(filename):
    return '.' in filename and \
        filename.rsplit('.',1)[1].lower() in ALLOWED_EXTENSIONS

@app.route("/" )
def hello():
    return render_template("index.html")

 #route to accept the video file 
@app.route("/submit", methods = ["POST","GET"])
def submit():
    global filename
    image_path = 'static/images/time_series.png'
    if request.method=="POST":
        #Checking if request has a file part
        if 'video' not in request.files:
            flash("No file part")
            return redirect(request.url)
        #Getting submitted file    
        file = request.files['video'] 
        #Checking if user submitted a file (The browser submits an empty file with no filename if the user doesn't select a file)
        if file.filename =="":
            flash("No selected file")
            return redirect(request.url)
        
        #checking if file has the allowed extension
        if file and allowed_file(file.filename):
            #cleaning up the filename and making sure if doesn't cause any dangerous operations in the server's file directory
            filename = secure_filename (file.filename)
            #saving file to uploads folder
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            #loading video into model
            flash('Processing submitted video',"submission")
            outputs, pulse_df=my_model.predict(filename)

            pulse_df['time'] = pd.to_datetime(pulse_df['time'], format = '%H:%M:%S  %d.%m.%Y')
            pulse_df.index = pulse_df['time']
            del pulse_df['time']
            
            fig, ax = plt.subplots()
            #plt.clf()
            sns.lineplot(pulse_df)
            #ax.set_xticklabels([t.get_text().split(".")[0] for t in ax.get_xticklabels()])
            ax.set_xticklabels([pd.to_datetime(t.get_text()).strftime('%H:%M:%S') for t in ax.get_xticklabels()])
            plt.ylabel('Pulse')
            plt.xlabel('time')
            plt.savefig(f'static/images/time_series.png')

            
            #video_path = 'static/videos/result.mp4'
            #return redirect(url_for('download_file',name=filename))
            return  render_template("submission.html",n={"filename":filename}, tables=[], image_path=image_path)
        
        #else if file type not allowed
        flash("File Type Not Allowed. \".mp4\",\".avi\",\".mkv\" only")
        return redirect(request.url)
     #TODO If user goes straight to /submit   
    return render_template("submission.html",tables=[], image_path=image_path)   


@app.route('/video')
def stream_video():
    return send_file('static/videos/result.mp4', mimetype='video/mp4')


if __name__ == "__main__":
    app.run(host='127.0.0.1',port=5000,debug=True,threaded=True)
