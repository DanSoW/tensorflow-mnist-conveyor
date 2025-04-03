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

def input_fn(file_pattern: List[str],
             data_accessor: DataAccessor,
             tf_transform_output: tft.TFTransformOutput,
             batch_size: int = 200) -> tf.data.Dataset:
    #Генерация входных данных для обучение / файнтюнинга

    return data_accessor.tf_dataset_factory(
            file_pattern,
            dataset_options.TensorFlowDatasetOptions(
                batch_size=batch_size, label_key=transformed_name(LABEL_KEY)),
            tf_transform_output.transformed_metadata.schema).repeat()

# Сборка модели DNN Keras для классификации цифр из MNIST
def build_keras_model() -> tf.keras.Model:
    model = tf.keras.Sequential()

    model.add(tf.keras.layers.InputLayer(input_shape=(784, ), name=transformed_name(IMAGE_KEY)))
    model.add(tf.keras.layers.Dense(64, activation='relu'))
    model.add(tf.keras.layers.Dropout(0.2))
    model.add(tf.keras.layers.Dense(64, activation='relu'))
    model.add(tf.keras.layers.Dropout(0.2))
    model.add(tf.keras.layers.Dense(10))
    
    model.compile(
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            optimizer=tf.keras.optimizers.RMSprop(learning_rate=0.0015),
            metrics=['sparse_categorical_accuracy'])
      
    model.summary(print_fn=absl.logging.info)

    return model



