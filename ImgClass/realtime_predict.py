import cv2
import numpy as np
import tensorflow as tf

classes = ['NG', 'OK']
loaded_model = tf.keras.models.load_model('dai-v3.h5')
# define a video capture object
vid = cv2.VideoCapture(1)

while(True):
    ret, frame = vid.read()
    start_col = (frame.shape[1]-frame.shape[0])//2
    end_col = frame.shape[1]-start_col
    cropped_frame = frame[:, start_col:end_col]
    resized_frame = cv2.resize(cropped_frame, (299,299))
    x = np.expand_dims(resized_frame, axis=0)
    images = np.vstack([x])

    pred = loaded_model.predict(images)
    pred1 = tf.nn.sigmoid(pred[0])
    pred2 = tf.where(pred1 < 0.5, 0, 1)

    if pred2 == 0:
        label = classes[0]
    else:
        label = classes[1]

    print(f'Prediction for the image: {pred1.numpy()}, the image is {label}')

    cv2.imshow('frame', resized_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
  
vid.release()
cv2.destroyAllWindows()