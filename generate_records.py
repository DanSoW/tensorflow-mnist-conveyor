import os
import tensorflow as tf
import numpy as np
import utils
import warnings

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.filterwarnings("ignore")

tf.executing_eagerly()
tf.config.run_functions_eagerly(True)

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

x_train = x_train.reshape(60000, 28, 28, 1).astype('float32') / 255.0
x_test  = x_test.reshape(10000, 28, 28, 1).astype('float32') / 255.0
y_train = y_train.reshape(60000, ).astype('int64')
y_test  = y_test.reshape(10000, ).astype('int64')

out_path_train = './train.tfrecord'
out_path_test  = './test.tfrecord'

# Создание tfrecord-файлов для обучающей и тестовой выборки
utils.convert_tfrecord(x_train, y_train, out_path_train)
utils.convert_tfrecord(x_test, y_test, out_path_test)



