import os
import tensorflow as tf
import numpy as np
import utils
import warnings

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.filterwarnings("ignore")

# Включаем режим "eager execution"
tf.executing_eagerly()
tf.config.run_functions_eagerly(True)

# Загрузка набора данных MNIST
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

x_train   = x_train.reshape(60000, 28, 28, 1).astype('float32')
#x_test    = x_test.reshape(10000, 28, 28, 1).astype('float32')
y_train   = y_train.reshape(60000, ).astype('int64')
#y_test    = y_test.reshape(10000, ).astype('int64')

# Определение пути к файлу train.tfrecord
out_path_train = './train.tfrecord'
#out_path_test  = './test.tfrecord'

# Создание tfrecord-файла
utils.convert_tfrecord(x_train, y_train, out_path_train)
#utils.convert_tfrecord(x_test, y_test, out_path_test)

# Функция для тестирования выходных данных из tfrecord-файла
@tf.function
def data_gen_output(filename):
    raw_dataset = tf.data.TFRecordDataset(filenames=[filename])

    for raw_record in raw_dataset.take(1):
        item = utils.convert_back(raw_record)
        return (item[0].shape, item[1].shape)

#data_gen_output(out_path_train)

