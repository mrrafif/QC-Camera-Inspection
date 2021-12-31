# source : https://stackoverflow.com/questions/65836699/flask-opencv-send-video-stream-to-extrenal-html-page

#library modbus
from pymodbus.client.sync import ModbusTcpClient
from time import sleep
#from random import uniform

#library camera
import threading
from flask import Flask, render_template, Response, make_response, send_from_directory, jsonify, request, session, copy_current_request_context
import cv2
import numpy as np
import json
from flask_socketio import SocketIO, emit 
from threading import Lock
import datetime
import base64
import tensorflow as tf
import time

# Setting Flask and socketio
async_mode = None
app = Flask(__name__, static_folder='public')
app.config['SECRET_KEY'] = 'secret!'
socketio = SocketIO(app, async_mode=async_mode)
thread = None
thread_lock = Lock()
countUser = 0

#init host dan port on modbus
host = '192.168.1.150'
port = 503

global client
client = ModbusTcpClient(host, port)
client.connect()
client.write_register(101, 1)


cameras = [
    cv2.VideoCapture(1, cv2.CAP_DSHOW)
]