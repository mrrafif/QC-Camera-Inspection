from _defect import *
from _color import *

@app.route('/tes')
def tes():
    """Video streaming home page."""
    response = make_response("try /video", 200)
    response.mimetype = "text/plain"
    return response

@app.route('/<path:filename>')
def staticPath(filename):
    if filename is None:
        filename = 'main.html'
    return send_from_directory(app.static_folder, filename)

@app.route('/')
def index():
    return send_from_directory(app.static_folder, 'main.html')


@socketio.on('connect')
def connect():
    global countUser
    countUser += 1
    print("USER CONNECTED, total user = {}".format(countUser))


@socketio.on('disconnect')
def disconnect():
    global countUser
    # global camera

    countUser -= 1
    #if countUser <= 0:
    #    camera.release()
    print("USER DISCONNECTED, total user = {}".format(countUser))


def run():
    app.run(host='0.0.0.0', debug=False)

if __name__ == '__main__':
    # creating threads
    t1 = threading.Thread(target=pred_frames, args=(0,))
    t3 = threading.Thread(target=color_frames, args=(0,))
    t2 = threading.Thread(target=run, args=())
  
    # starting threads
    t1.start()
    t3.start()
    t2.start()