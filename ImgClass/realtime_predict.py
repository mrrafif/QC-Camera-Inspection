import cv2
import numpy as np
import tensorflow as tf

# classes_bin = ['NG', 'OK']
# model_bin = tf.keras.models.load_model('dai-v3.h5')

classes_multi = ['NGFL', 'NGSC' ,'NGSH', 'OK']
model_multi = tf.keras.models.load_model('dai-v4.h5')

vid = cv2.VideoCapture(1)

while(True):
    ret, frame = vid.read()
    start_col = (frame.shape[1]-frame.shape[0])//2
    end_col = frame.shape[1]-start_col
    cropped_frame = frame[:, start_col:end_col]
    resized_frame = cv2.resize(cropped_frame, (299,299))
    x = np.expand_dims(resized_frame, axis=0)
    images = np.vstack([x])

    # FOR BINARY CLASSIFICATION
    # pred_bin = model_bin.predict(images)
    # sigmoid_bin = tf.nn.sigmoid(pred_bin[0]).numpy()
    # sigmoid_bin01 = tf.where(sigmoid_bin < 0.5, 0, 1)
    # if sigmoid_bin01 == 0:
    #   label_bin = classes_bin[0]
    # else:
    #   label_bin = classes_bin[1]

    # FOR MULTI CLASS CLASSIFICATION
    pred_multi = model_multi.predict(images)
    sigmoid_multi = tf.nn.sigmoid(pred_multi[0]).numpy()
    for i in range(len(sigmoid_multi)):
        if sigmoid_multi[i] == max(sigmoid_multi):
            sigmoid_multi_max = sigmoid_multi[i]
            label_multi = classes_multi[i]
            class_number = i
            break

    print(f'Sigmoid result: {sigmoid_multi} - Prediction: {label_multi} - Confidence: {round(sigmoid_multi_max*100, 3)} %')
    # print(f'Sigmoid result: {sigmoid_bin} - Prediction: {label_bin}')

    cv2.imshow('frame', resized_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
  
vid.release()
cv2.destroyAllWindows()