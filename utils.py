import tensorflow as tf

# Функции обёртки для формирования данных для элементов форм
def wrap_int64(val):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[val]))

def wrap_bytes(val):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[val]))

def wrap_float(val):
    return tf.train.Feature(float_list=tf.train.FloatList(value=val))

# Запись данных в файл record
@tf.function
def convert_tfrecord(images, labels, out_path):
    with tf.io.TFRecordWriter(out_path) as writer:
        for image, label in zip(images, labels):
            img = image.reshape((784, ))

            # Определение формата записи в TFRecord
            mnist = {
                'image_class': wrap_int64(int(label)),
                'image_floats': wrap_float(img)
            }

            feature = tf.train.Features(feature=mnist)
            example = tf.train.Example(features=feature)

            # Сериализация данных в строку
            serialized = example.SerializeToString()

            # Запись элемента в TFRecord
            writer.write(serialized)

# Парсинз из строки TFRecord
@tf.function
def convert_back(serialized):
    # Форма для парсинга сериализованной строки
    feature = {
            'image_class': tf.io.VarLenFeature(tf.int64),
            'image_floats': tf.io.FixedLenFeature((784, ), tf.float32)
    }

    # Парсинг одного экземпляра записи TFRecord
    parsed_example = tf.io.parse_single_example(serialized=serialized, features=feature)

    image = parsed_example['image_floats']
    img = tf.reshape(image, shape=[28, 28, 1])
    
    label = parsed_example['image_class']

    return img, label




