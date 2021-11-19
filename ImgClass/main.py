import matplotlib.pyplot as plt
import numpy as np
import os
import tensorflow as tf
from keras.preprocessing import image

classes = ['NG', 'OK']
loaded_model = tf.keras.models.load_model('dai-v3.h5')

# test_dataset = tf.keras.utils.image_dataset_from_directory(
#     'workspace/images/test',
#     shuffle=True,
#     image_size=(299,299))
# image_batch, label_batch = test_dataset.as_numpy_iterator().next()
# predictions = loaded_model.predict_on_batch(image_batch).flatten()
# predictions = tf.nn.sigmoid(predictions)
# predictions = tf.where(predictions < 0.5, 0, 1)

img = image.load_img('testingok2.jpeg', target_size=(299, 299))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
images = np.vstack([x])

pred = loaded_model.predict(images)
pred1 = tf.nn.sigmoid(pred[0])
pred2 = tf.where(pred1 < 0.5, 0, 1)

if pred2 == 0:
  label = classes[0]
else:
  label = classes[1]

# print('Predictions Batch:\n', predictions.numpy())
# print('Labels Batch:\n', label_batch)
print(f'Prediction for the image: {pred2.numpy()}, the image is {label}')

# plt.figure(figsize=(10, 10))
# for i in range(9):
#   ax = plt.subplot(3, 3, i + 1)
#   plt.imshow(image_batch[i].astype('uint8'))
#   plt.title(classes[predictions[i]])
#   plt.axis('off')
# plt.show()

plt.figure()
plt.axis('off')
plt.title(label)
result = plt.imshow(img)
plt.show()