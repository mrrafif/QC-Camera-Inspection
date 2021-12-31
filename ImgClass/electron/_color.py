from _declare import *

value = ['']
# global isGood
# isGood = {
#     1: False,
#     2: False
# }

def color_frames(idCamera):
    # global isGood
    while True:
        cameraId = int(idCamera)
        _, img = cameras[cameraId].read()

        #converting frame(img) from BGR (Blue-Green-Red) to HSV (hue-saturation-value)
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        
        #defining the range of Yellow color
        # yel_lower = np.array([25, 50, 50],np.uint8)
        # yel_upper = np.array([50, 255, 255],np.uint8)
        yel_lower = np.array([5, 125, 125],np.uint8)
        yel_upper = np.array([50, 255, 255],np.uint8)
        yel_mask = cv2.inRange(hsv, yel_lower, yel_upper)

        #Morphological transformation, Dilation         
        res=cv2.bitwise_and(img, img, mask = yel_mask)

        #Tracking Colour (Yellow) 
        (contours,hierarchy)=cv2.findContours(yel_mask,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
        
        # isGood[cameraId] = False
        for pic, contour in enumerate(contours):
                area = cv2.contourArea(contour)
                if(area>150):
                        x,y,w,h = cv2.boundingRect(contour)     
                        img = cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),3)
                        # isGood[cameraId] = ((255,0,0) in img)
                        value[cameraId] = 'Kuning'
                        #write nilai register
                        client.write_register(102, 0)
                        #time.sleep(5)
        value[cameraId] = 'Gak Kuning'
        client.write_register(102, 1)
        global responseFrame
        responseFrame = img

def feed_frames_color(idCamera): #stream video ke web
    while True:
        cameraId = int(idCamera)
        _, frame = cameras[cameraId].read()
        if not _:
            break
        else:
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()

            yield (b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/color/video/<idCamera>')
def video_feed_color(idCamera):
    #Video streaming route. Put this in the src attribute of an img tag
    return Response(feed_frames_color(idCamera), mimetype='multipart/x-mixed-replace; boundary=frame')

@socketio.on('getStatus', namespace='/ws/color')
def test_broadcast_message_color():
    while True:
        emit('message', {'data': value}, broadcast=True)
        
        now = datetime.datetime.now()
        later = now + datetime.timedelta(0,1)
        while now <= later:
            now = datetime.datetime.now()
