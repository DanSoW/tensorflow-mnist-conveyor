from typing import List

import absl
import tensorflow as tf
import tensorflow_transform as tft
from tfx.components.trainer.fn_args_utils import DataAccessor
from tfx_bsl.tfxio import dataset_options

IMAGE_KEY = 'image_floats'
LABEL_KEY = 'image_class'

def transformed_name(key):
    return key + '_xf'

# Определение источника входных данных для обучения модели
def input_fn(file_pattern: List[str],
             data_accessor: DataAccessor,
             tf_transform_output: tft.TFTransformOutput,
             batch_size: int = 200) -> tf.data.Dataset:
    
    # Создание фабрики для доступа к набору данных для обучения или валидации
    return data_accessor.tf_dataset_factory(
            file_pattern,
            dataset_options.TensorFlowDatasetOptions(
                batch_size=batch_size, label_key=transformed_name(LABEL_KEY)),
            tf_transform_output.transformed_metadata.schema).repeat()


# Сборка модели нейронной сети для классификации цифр из MNIST
def build_keras_model() -> tf.keras.Model:
    model = tf.keras.Sequential()

    model.add(tf.keras.layers.InputLayer(shape=(784, ), dtype=tf.float32, name=transformed_name(IMAGE_KEY)))
    model.add(tf.keras.layers.Dense(64, activation='relu'))
    model.add(tf.keras.layers.Dropout(0.2))
    model.add(tf.keras.layers.Dense(64, activation='relu'))
    model.add(tf.keras.layers.Dropout(0.2))
    model.add(tf.keras.layers.Dense(10))
    
    # Компиляция модели
    model.compile(
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            optimizer=tf.keras.optimizers.RMSprop(learning_rate=0.0015),
            metrics=['sparse_categorical_accuracy'])
      
    # Суммаризация
    model.summary(print_fn=absl.logging.info)

    return model

# Обработка входных данных (используется в Transform)
def preprocessing_fn(inputs):
    outputs = {}

    # Нормализация значений признаков
    outputs[transformed_name(IMAGE_KEY)] = inputs[IMAGE_KEY] / 255.0
    outputs[transformed_name(LABEL_KEY)] = inputs[LABEL_KEY]

    return outputs
    



