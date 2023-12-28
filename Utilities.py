import joblib
import numpy as np
import tensorflow as tf

class Utils:
  def __init__(self):
    self.load_models()
    self.IMG_SIZE = (224, 224)
    self.class_names = ['Normal', 'Tuberculosis']
  
  def load_models(self):
    self.cnn = tf.keras.models.load_model('Models/CNN')
    self.dnn = tf.keras.models.load_model("Models/DNN")
    self.knn = joblib.load("Models/KNN.sav")
    self.regressor = joblib.load("Models/regressor.sav")
    self.dt = joblib.load("Models/DTC.sav")
    self.hybrid = Hybrid_Model(self.dnn, self.cnn, self.regressor)

  def load_and_prep(self, filepath):
    img = tf.io.read_file(filepath)
    img = tf.io.decode_image(img)
    img = tf.image.resize(img, self.IMG_SIZE)
    if img.shape[2] == 1:
      img = tf.image.grayscale_to_rgb(img)

    return img
  
  def make_predictions(self, model_name, img_path):
    img = self.load_and_prep(img_path)
    img_arr = np.array(img)/255

    if model_name == 'RFC':
      pred = self.regressor.predict(img_arr.reshape(1, 150528))[0]
      prob = self.regressor.predict_proba(img_arr.reshape(1, 150528))
    elif model_name == 'KNN':
      pred = self.knn.predict(img_arr.reshape(1, 150528))[0]
      prob = self.knn.predict_proba(img_arr.reshape(1, 150528))
    elif model_name == 'DT':
      pred = self.dt.predict(img_arr.reshape(1, 150528))[0]
      prob = self.dt.predict_proba(img_arr.reshape(1, 150528))
    elif model_name == 'CNN':
      with tf.device('/cpu:0'):
        prob = self.cnn.predict(tf.expand_dims(img, axis=0))
        pred = prob[0].argmax()
    elif model_name == 'DNN':
      with tf.device('/cpu:0'):
        prob = self.dnn.predict(tf.expand_dims(img, axis=0))
        pred = prob[0].argmax()
    elif model_name == 'hybrid':
      pred = self.hybrid.predict(img)
      label = self.class_names[pred]
      return 100, label

    label = self.class_names[pred]
    prob = round(max(prob[0]), 2) * 100

    return int(prob), label

# Updated to reduce latency
class Hybrid_Model:
  def __init__(self, dnn, cnn, regressor):
    self.img = None
    self.dnn = dnn
    self.cnn = cnn
    self.regressor = regressor

  def predict(self, img):
    self.img = img
    with tf.device('/cpu:0'):
        pred_1 = self.cnn.predict(tf.expand_dims(img, axis=0))[0].argmax()
        pred_2 = self.dnn.predict(tf.expand_dims(img, axis=0))[0].argmax()
    img = np.array(img)/255
    pred_3 = self.regressor.predict(img.reshape(1, 150528))[0]

    preds = np.array([pred_1, pred_2, pred_3])
    max_pred = np.argmax(np.bincount(preds))

    return max_pred

if __name__ == '__main__':
  util = Utils()
  img_path = 'Test/tub-21.jpg'
  s, y = util.make_predictions('RFC', img_path)
  print(s, y)