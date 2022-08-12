from flask import Flask, render_template, Response, jsonify,request
from camera import VideoCamera
import cv2

app = Flask(__name__)

video_stream = VideoCamera()

@app.route('/')
def index():
    return render_template('index.html')

def gen(camera):
    while(True):
        frame = camera.get_frame()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

@app.route('/video_feed')
def video_feed():
        return Response(gen(video_stream),
                    mimetype='multipart/x-mixed-replace; boundary=frame')
@app.route('/tasks')
def tasks():
    x=video_stream.getEmotion()
    if(x=='happiness'):
      return render_template('index1.html',message=x)
    elif(x=='anger'):
        return render_template('index2.html',message=x)
    elif(x=='fear'):
        return render_template('index3.html',message=x)
    elif(x=='disgust'):
        return render_template('index4.html',message=x)
    elif(x=='neutral'):
        return render_template('index5.html',message=x)
    elif(x=='surprise'):
        return render_template('index6.html',message=x)
    else:
        return render_template('index7.html',message=x)

@app.route('/tasks1')
def tasks1():
    return render_template('capture.html')

def gen1(camera):
    frame = camera.get_frame1()
    yield (b'--frame\r\n'
            b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

@app.route('/tasks2')
def tasks2():
    return Response(gen1(video_stream),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(host='127.0.0.1', debug=True,port="5000")

