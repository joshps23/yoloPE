from flask import Flask, render_template, Response,jsonify,request,session,redirect, send_file
from dotenv import load_dotenv
from pathlib import Path

#FlaskForm--> it is required to receive input from the user
# Whether uploading a video file  to our object detection model

from flask_wtf import FlaskForm


from wtforms import FileField, SubmitField,StringField,DecimalRangeField,IntegerRangeField
from werkzeug.utils import secure_filename
from wtforms.validators import InputRequired,NumberRange
import os


# Required to run the YOLOv8 model
import cv2

# YOLO_Video is the python file which contains the code for our object detection model
#Video Detection is the Function which performs Object Detection on Input Video
from YOLO_Video import video_detection
from YOLO_Video_Class import video_classify
from YOLO_lop_web import lop_detection
app = Flask(__name__)

load_dotenv()

app.config['SECRET_KEY'] = os.environ["SECRET_KEY"]
app.config['UPLOAD_FOLDER'] = 'static/files'


#Use FlaskForm to get input video file  from user

class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y

class ThreadedCamera(object):
    def __init__(self, src=0):
        self.capture = cv2.VideoCapture(src)
        self.capture.set(cv2.CAP_PROP_BUFFERSIZE, 2)
       
        # FPS = 1/X
        # X = desired FPS
        self.FPS = 1/30
        self.FPS_MS = int(self.FPS * 1000)
        
        # Start frame retrieval thread
        self.thread = Thread(target=self.update, args=())
        self.thread.daemon = True
        self.thread.start()
        
    def update(self):
        while True:
            if self.capture.isOpened():
                (self.status, self.frame) = self.capture.read()
            time.sleep(self.FPS)
            
    def show_frame(self):
        cv2.imshow('frame', self.frame)
        cv2.waitKey(self.FPS_MS)

class UploadFileForm(FlaskForm):
    #We store the uploaded video file path in the FileField in the variable file
    #We have added validators to make sure the user inputs the video in the valid format  and user does upload the
    #video when prompted to do so
    file = FileField("File",validators=[InputRequired()])
    submit = SubmitField("Run")


def generate_frames(path_x = '', mode = '', path_dl = ''):
    yolo_output = video_detection(path_x,mode,path_dl)
    for detection_ in yolo_output:
        ref,buffer=cv2.imencode('.jpg',detection_)

        frame=buffer.tobytes()
        yield (b'--frame\r\n'
                    b'Content-Type: image/jpeg\r\n\r\n' + frame +b'\r\n')

def generate_lop(path_x = '', path_dl = ''):
    yolo_output = lop_detection(path_x,path_dl)
    for detection_ in yolo_output:
        ref,buffer=cv2.imencode('.jpg',detection_)

        frame=buffer.tobytes()
        yield (b'--frame\r\n'
                    b'Content-Type: image/jpeg\r\n\r\n' + frame +b'\r\n')


def generate_classimg(path_x = ''):
    yolo_output = video_classify(path_x)
    for detection_ in yolo_output:
        ref,buffer=cv2.imencode('.jpg',detection_)

        frame=buffer.tobytes()
        yield (b'--frame\r\n'
                    b'Content-Type: image/jpeg\r\n\r\n' + frame +b'\r\n')

def generate_frames_web(path_x):
    yolo_output = video_detection(path_x)
    for detection_ in yolo_output:
        ref,buffer=cv2.imencode('.jpg',detection_)

        frame=buffer.tobytes()
        yield (b'--frame\r\n'
                    b'Content-Type: image/jpeg\r\n\r\n' + frame +b'\r\n')

# @app.route('/', methods=['GET','POST'])
@app.route('/home', methods=['GET','POST'])
# def home():
#     session.clear()
#     return render_template('index.html')
def home():
    session.clear()
    form = UploadFileForm()
    if form.validate_on_submit():
        # Our uploaded video file path is saved here
        file = form.file.data
        file.save(os.path.join(os.path.abspath(os.path.dirname(__file__)), app.config['UPLOAD_FOLDER'],
                               secure_filename(file.filename)))  # Then save the file
        # Use session storage to save video file path
        session['video_path'] = os.path.join(os.path.abspath(os.path.dirname(__file__)), app.config['UPLOAD_FOLDER'],
                                             secure_filename(file.filename))
    return render_template('videoprojectnew.html', form=form)




@app.route('/2rs', methods=['GET','POST'])
def terra():
    # Upload File Form: Create an instance for the Upload File Form
    
    form = UploadFileForm()
    if form.validate_on_submit():
        # Our uploaded video file path is saved here

        file = form.file.data
        file.save(os.path.join(os.path.abspath(os.path.dirname(__file__)), app.config['UPLOAD_FOLDER'],
                               secure_filename(file.filename)))  # Then save the file
        # Use session storage to save video file path
        session['video_path'] = os.path.join(os.path.abspath(os.path.dirname(__file__)), app.config['UPLOAD_FOLDER'],
                                             secure_filename(file.filename))
    return render_template('terra.html', form=form)

@app.route('/lop', methods=['GET','POST'])
def lop():
    # Upload File Form: Create an instance for the Upload File Form
    
    form = UploadFileForm()
    if form.validate_on_submit():
        # Our uploaded video file path is saved here

        file = form.file.data
        file.save(os.path.join(os.path.abspath(os.path.dirname(__file__)), app.config['UPLOAD_FOLDER'],
                               secure_filename(file.filename)))  # Then save the file
        # Use session storage to save video file path
        session['video_path'] = os.path.join(os.path.abspath(os.path.dirname(__file__)), app.config['UPLOAD_FOLDER'],
                                             secure_filename(file.filename))
    return render_template('lop.html', form=form)


# Rendering the Webcam Rage
#Now lets make a Webcam page for the application
#Use 'app.route()' method, to render the Webcam page at "/webcam"
@app.route("/webcam", methods=['GET','POST'])



def webcam():
    session.clear()
    return render_template('ui.html')
# @app.route('/FrontPage', methods=['GET','POST'])
@app.route('/', methods=['GET','POST'])
def front():
    # Upload File Form: Create an instance for the Upload File Form
    
    form = UploadFileForm()
    if form.validate_on_submit():
        # Our uploaded video file path is saved here

        file = form.file.data
        file.save(os.path.join(os.path.abspath(os.path.dirname(__file__)), app.config['UPLOAD_FOLDER'],
                               secure_filename(file.filename)))  # Then save the file
        # Use session storage to save video file path
        session['video_path'] = os.path.join(os.path.abspath(os.path.dirname(__file__)), app.config['UPLOAD_FOLDER'],
                                             secure_filename(file.filename))
    return render_template('videoprojectnew.html', form=form)
@app.route('/ball', methods=['GET','POST'])
def ballform():
    # Upload File Form: Create an instance for the Upload File Form
    
    form = UploadFileForm()
    if form.validate_on_submit():
        # Our uploaded video file path is saved here

        file = form.file.data
        file.save(os.path.join(os.path.abspath(os.path.dirname(__file__)), app.config['UPLOAD_FOLDER'],
                               secure_filename(file.filename)))  # Then save the file
        # Use session storage to save video file path
        session['video_path'] = os.path.join(os.path.abspath(os.path.dirname(__file__)), app.config['UPLOAD_FOLDER'],
                                             secure_filename(file.filename))
    return render_template('videoball.html', form=form)

@app.route('/download', methods=['GET', 'POST'])
def downloadFile (filename=''):
    #For windows you need to use drive name [ex: F:/Example.pdf]
    session_path = session['video_path']
    uploads_folder = "uploads"
    
    dir_path = os.path.dirname(os.path.realpath(__file__))
    filename=os.path.basename(session_path)
    filename=Path(filename).stem
    uploads_path = os.path.join(dir_path, app.config['UPLOAD_FOLDER'],filename)
    uploads_path = uploads_path + '_analysed.mp4'
    # print(f"Filename is {os.path.basename(session_path)}")
    # print(f"Directory is {dir_path}")
    # print(f"Download Path is {uploads_path}")
    path = 'output.avi'
    return send_file(uploads_path, as_attachment=True)


@app.route('/video')
def video():
    session_path = session.get('video_path', None)
    uploads_folder = "uploads"
    
    dir_path = os.path.dirname(os.path.realpath(__file__))
    filename=os.path.basename(session_path)
    filename=Path(filename).stem
    uploads_path = os.path.join(dir_path, app.config['UPLOAD_FOLDER'], filename)
    uploads_path = uploads_path + '_analysed.mp4'
    #return Response(generate_frames(path_x='static/files/bikes.mp4'), mimetype='multipart/x-mixed-replace; boundary=frame')
    return Response(generate_frames(path_x = session.get('video_path', None), mode = "space", path_dl = uploads_path),mimetype='multipart/x-mixed-replace; boundary=frame')
@app.route('/videoball')
def videoball():
    #return Response(generate_frames(path_x='static/files/bikes.mp4'), mimetype='multipart/x-mixed-replace; boundary=frame')
    session_path = session.get('video_path', None)
    uploads_folder = "uploads"
    
    dir_path = os.path.dirname(os.path.realpath(__file__))
    filename=os.path.basename(session_path)
    filename=Path(filename).stem
    uploads_path = os.path.join(dir_path, app.config['UPLOAD_FOLDER'], filename)
    uploads_path = uploads_path + '_analysed.mp4'
    #return Response(generate_frames(path_x='static/files/bikes.mp4'), mimetype='multipart/x-mixed-replace; boundary=frame')
    return Response(generate_frames(path_x = session.get('video_path', None), mode = "ball", path_dl = uploads_path),mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/detect_lop')
def detect_lop():
    session_path = session.get('video_path', None)
    uploads_folder = "uploads"
    
    dir_path = os.path.dirname(os.path.realpath(__file__))
    filename=os.path.basename(session_path)
    filename=Path(filename).stem
    uploads_path = os.path.join(dir_path, app.config['UPLOAD_FOLDER'], filename)
    uploads_path = uploads_path + '_analysed.mp4'
    #return Response(generate_frames(path_x='static/files/bikes.mp4'), mimetype='multipart/x-mixed-replace; boundary=frame')
    return Response(generate_lop(path_x = session.get('video_path', None), path_dl = uploads_path),mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/classify')
def classify():
    session_path = session.get('video_path', None)
    uploads_folder = "uploads"
    

    #return Response(generate_frames(path_x='static/files/bikes.mp4'), mimetype='multipart/x-mixed-replace; boundary=frame')
    return Response(generate_classimg(path_x = session.get('video_path', None)),mimetype='multipart/x-mixed-replace; boundary=frame')

# To display the Output Video on Webcam page
@app.route('/webapp')
def webapp():
    #return Response(generate_frames(path_x = session.get('video_path', None),conf_=round(float(session.get('conf_', None))/100,2)),mimetype='multipart/x-mixed-replace; boundary=frame')
    return Response(generate_frames_web(path_x=0), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route("/clearsession")
def clear():
    session.clear()
    return redirect("/")
@app.route("/clearlop")
def clearlop():
    session.clear()
    return redirect("/lop")

@app.route("/clearterra")
def clearterra():
    session.clear()
    return redirect("/2rs")
@app.route("/clearball")
def clearball():
    session.clear()
    return redirect("/ball")

if __name__ == "__main__":

    app.run(debug=True,host="0.0.0.0",port=5000)