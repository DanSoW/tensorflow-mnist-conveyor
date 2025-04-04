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

x_train   = x_train.reshape(60000, 28, 28, 1).astype('float32')
x_test    = x_test.reshape(10000, 28, 28, 1).astype('float32')
y_train   = y_train.reshape(60000, ).astype('int64')
y_test    = y_test.reshape(10000, ).astype('int64')

out_path_train = './train.tfrecord'
out_path_test  = './test.tfrecord'

# Создание tfrecord-файлов для обучающей и тестовой выборки
utils.convert_tfrecord(x_train, y_train, out_path_train)
utils.convert_tfrecord(x_test, y_test, out_path_test)

@tf.function
def data_gen2(filename):
    raw_dataset = tf.data.TFRecordDataset(filenames=[filename])

    for raw_record in raw_dataset.take(1):
        example = tf.train.Example()
        example.ParseFromString(raw_record.numpy())
        print(raw_record)
        print(example)
        return utils.convert_back(raw_record)

#data_gen2(out_path_train)

