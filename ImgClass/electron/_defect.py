from _declare import *

camValues = ['']

MODEL_PATH = ['dai-v3.h5', 'dai-v4.h5', 'dai-v5-bin.h5']
model_bin = tf.keras.models.load_model(MODEL_PATH[0])
model_multi = tf.keras.models.load_model(MODEL_PATH[1])

def binary(model, frame):
    global label_bin
    classes_bin = ['NG', 'OK']
    start_col = (frame.shape[1]-frame.shape[0])//2
    end_col = frame.shape[1]-start_col
    cropped_frame = frame[:, start_col:end_col]
    resized_frame = cv2.resize(cropped_frame, (299,299))
    x = np.expand_dims(resized_frame, axis=0)
    images = np.vstack([x])

    pred_bin = model.predict(images)
    sigmoid_bin = tf.nn.sigmoid(pred_bin[0]).numpy()
    sigmoid_bin01 = tf.where(sigmoid_bin < 0.5, 0, 1) #0: NG, 1: OK
    if sigmoid_bin01 == 0:
        label_bin = classes_bin[0]
        client.write_register(102, 1)
    else:
        label_bin = classes_bin[1]
        client.write_register(102, 0)

def multiclass(model, frame):
    global label_multi
    classes_multi = ['NG', 'NG' ,'NG', 'OK'] #['NGFL', 'NGSC' ,'NGSH', 'OK']
    start_col = (frame.shape[1]-frame.shape[0])//2
    end_col = frame.shape[1]-start_col
    cropped_frame = frame[:, start_col:end_col]
    resized_frame = cv2.resize(cropped_frame, (299,299))
    x = np.expand_dims(resized_frame, axis=0)
    images = np.vstack([x])

    pred_multi = model.predict(images)
    sigmoid_multi = tf.nn.sigmoid(pred_multi[0]).numpy()
    for i in range(len(sigmoid_multi)):
        if sigmoid_multi[i] == max(sigmoid_multi):
            sigmoid_multi_max = sigmoid_multi[i]
            label_multi = classes_multi[i]
            class_number = i
            if class_number == 3:
                client.write_register(102, 0)
                break
            else:
                client.write_register(102, 1)
                break  

def pred_frames(idCamera): #ngasih predict NG/OK
    while True:
        cameraId = int(idCamera)
        _, frame = cameras[cameraId].read()
        binary(model_bin, frame)
        # multiclass(model_multi, frame)
        camValues[cameraId] = label_bin
        # time.sleep(2)

def feed_frames(idCamera): #stream video ke web
    while True:
        cameraId = int(idCamera)
        _, frame = cameras[cameraId].read()
        start_col = (frame.shape[1]-frame.shape[0])//2
        end_col = frame.shape[1]-start_col
        cropped_frame = frame[:, start_col:end_col]
        resized_frame = cv2.resize(cropped_frame, (480,480))

        if not _:
            break
        else:
            ret, buffer = cv2.imencode('.jpg', resized_frame)
            frame = buffer.tobytes()

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/defect/video/<idCamera>')
def video_feed_defect(idCamera):
    #Video streaming route. Put this in the src attribute of an img tag
    return Response(feed_frames(idCamera), mimetype='multipart/x-mixed-replace; boundary=frame')

@socketio.on('getStatus', namespace='/ws/defect')
def test_broadcast_message_defect():
    while True:
        emit('message', {'data': camValues}, broadcast=True)
        
        now = datetime.datetime.now()
        later = now + datetime.timedelta(milliseconds=100)
        while now <= later:
            now = datetime.datetime.now()