import requests
import json
import tensorflow as tf
import numpy as np

tf.executing_eagerly()
tf.config.run_functions_eagerly(True)

tfrecord_path = "...tensorflow-tfx\\client-api\\train.tfrecord"

# Парсинг из tfrecord
@tf.function
def convert_back(serialized):
  # Форма для парсинга входных данных
  feature = {
      'image_class' : tf.io.VarLenFeature(tf.int64),
      'image_floats' : tf.io.FixedLenFeature((784, ), tf.float32)
  }

  # Парсинг одного экземпляра данных
  parsed_example = tf.io.parse_single_example(serialized=serialized, features=feature)

  image = parsed_example['image_floats']
  label = parsed_example['image_class']

  return image, label

# Получение 10-ти спарсенных записей из tfrecord-файла
@tf.function
def data_gen_output(filename):
  raw_dataset = tf.data.TFRecordDataset(filenames=[filename])

  data = []

  for raw_record in raw_dataset.take(10):
    data.append(convert_back(raw_record))

  return data

items = data_gen_output(tfrecord_path)

# Заголовок для HTTP-запроса
headers = {"content-type": "application/json"}

for i in range(len(items)):
    img, label = items[i]
    img = img.numpy()

    data = {
        # Определение сигнатуры
        "signature_name": "serving_default",
        # Определение входных данных для API запроса
        "instances": [img.tolist()]
    }

    # Отправка запроса серверу
    json_response = requests.post('http://localhost:8501/v1/models/saved_model:predict', data=json.dumps(data), headers=headers)

    # Получение массива предсказанных значений
    predictions = json.loads(json_response.text)['predictions']

    # Определение класса, к которому относится цифра на изображении
    defClass = int(np.argmax(predictions))

    print("Predict: ", defClass)
    print("Fact: ", label.values.numpy()[0])
    print()
