import tensorflow as tf

# Схема для хранения меток из набора данных
def wrap_int64(val):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[val]))

# Схема для хранения float-значений признаков
def wrap_float(val):
    return tf.train.Feature(float_list=tf.train.FloatList(value=val))

# Запись набора данных в tfrecord-файл
@tf.function
def convert_tfrecord(images, labels, out_path):
    # Открытие потока для записи в tfrecord-файл
    with tf.io.TFRecordWriter(out_path) as writer:
        # Упаковка элементов набора данных и проход по нему через цикл
        for image, label in zip(images, labels):
            # Изменение размера тензора (процедура flatten)
            img = image.reshape((784, ))

            # Определение элемента одной записи
            mnist = {
                'image_class': wrap_int64(int(label)),
                'image_floats': wrap_float(img)
            }

            # Определение данных и формирование элемента прототипа (Example)
            feature = tf.train.Features(feature=mnist)
            example = tf.train.Example(features=feature)

            # Сериализация прототипа в строку
            serialized = example.SerializeToString()

            # Запись элемента в tfrecord-файл (построчно)
            writer.write(serialized)

# Парсинг одного элемента из tfrecord-файла
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
    # image = tf.reshape(image, shape=[28, 28, 1])
    label = parsed_example['image_class']

    return image, label




