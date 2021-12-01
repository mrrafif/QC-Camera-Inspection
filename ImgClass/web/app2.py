# Untuk REST API
from flask import Flask, render_template, Response, make_response, send_from_directory, jsonify, request, session, copy_current_request_context

# Untuk ngirim pesan dengan socketio
from flask_socketio import SocketIO, emit, disconnect

# Untuk capture gambar
import cv2
from numpy.core.fromnumeric import resize

# TensorFlow dan tf.keras untuk AI
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image as keras_image
from tensorflow.keras.layers import Reshape
import tensorflow as tf

# Opencv untuk menangkap gambar dari kamera
import cv2
from PIL import Image

# Numpy untuk pemrosesan matriks
import numpy as np

# utility
import re
import base64
import os
import sys
import json
import datetime
import threading

async_mode = None
app = Flask(__name__, static_folder='public')
app.config['SECRET_KEY'] = 'secret!'
socketio = SocketIO(app, async_mode=async_mode)

countUser = 0

cameras = [
    cv2.VideoCapture(1)
]
# cameras[0].set(cv2.CAP_PROP_FRAME_WIDTH, 500)
# cameras[0].set(cv2.CAP_PROP_FRAME_HEIGHT, 500)

# Load model
MODEL_PATH = 'dai-v3.h5'
model = load_model(MODEL_PATH)
print('Model loaded. Start serving...')

# value = ['ba', 'ca', 'da', 'dha', 'ga', 'ha', 'ja', 'ka', 'la', 'ma',
#         'na', 'nga', 'nya', 'pa', 'ra', 'sa', 'ta', 'tha', 'wa', 'ya']

value = ['NG', 'OK']

n_size = 100

camFrames = [False]
camStatus = [False]

camValues = [""]

def genFrames(cameraId):
    global camValues
    global camFrames
    global camStatus
    
    while True:
        _, frame = cameras[cameraId].read()
        start_col = (frame.shape[1]-frame.shape[0])//2
        end_col = frame.shape[1]-start_col
        cropped_frame = frame[:, start_col:end_col]
        resized_frame = cv2.resize(cropped_frame, (299,299))
        x = np.expand_dims(resized_frame, axis=0)
        images = np.vstack([x])

        pred = model.predict(images)
        pred1 = tf.nn.sigmoid(pred[0])
        pred2 = tf.where(pred1 < 0.5, 0, 1)

        if pred2 == 0:
            script = value[0]
        else:
            script = value[1]
        # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # Convert the captured frame
        # im = Image.fromarray(frame)

        #Resizing into dimensions you used while training
        # im = im.resize((100,100))
        # x = np.array(im)
        # x = np.float32(x)
        # x = np.reshape(x, (-1, 100))
        # print(len(x[0]))

        # x = np.expand_dims(x, axis=2)
        # x = np.expand_dims(x, axis=0)

        # x = np.vstack([x])
        # print ("data prep stage")
        # out = model.predict(x)
        # script = value[np.argmax(out)]
      
        camValues[cameraId] = script
        camFrames[cameraId] = frame
        camStatus[cameraId] = _

def sendFrame(idCamera):
    cameraId = int(idCamera)
    while True:
        success, frame = cameras[cameraId].read()
        start_col = (frame.shape[1]-frame.shape[0])//2
        end_col = frame.shape[1]-start_col
        cropped_frame = frame[:, start_col:end_col]
        resized_frame = cv2.resize(cropped_frame, (480,480))
        if not success:
            break
        else:
            ret, buffer = cv2.imencode('.jpg', resized_frame)
            resized_frame = buffer.tobytes()

            yield (b'--frame\r\n'
                    b'Content-Type: image/jpeg\r\n\r\n' + resized_frame + b'\r\n')

@app.route('/video/<idCamera>')
def video_feed(idCamera):
    #Video streaming route. Put this in the src attribute of an img tag
    return Response(sendFrame(idCamera), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/')
def index():
    return send_from_directory(app.static_folder, 'index.html')

# Broadcast status every seconds
@socketio.on('getStatus', namespace='/ws')
def test_broadcast_message():
    while True:
        emit('message', {'data': camValues}, broadcast=True)
        
        now = datetime.datetime.now()
        later = now + datetime.timedelta(0,1)
        while now <= later:
            now = datetime.datetime.now()


@socketio.on('connect')
def connect():
    global countUser
    countUser += 1
    print("USER CONNECTED, total user = {}".format(countUser))


@socketio.on('disconnect')
def disconnect():
    global countUser
    countUser -= 1
    print("USER DISCONNECTED, total user = {}".format(countUser))

def run():
    app.run(host='0.0.0.0', debug=False)

if __name__ == '__main__':
    # creating threads
    t1 = threading.Thread(target=genFrames, args=(0,))
    t2 = threading.Thread(target=run, args=())
  
    # starting threads
    t1.start()
    t2.start()
  