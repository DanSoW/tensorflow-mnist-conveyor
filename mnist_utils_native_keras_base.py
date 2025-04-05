from typing import List

import absl
import tensorflow as tf
import tensorflow_transform as tft
from tfx.components.trainer.fn_args_utils import DataAccessor
from tfx_bsl.tfxio import dataset_options
from absl import logging

IMAGE_KEY = 'image_floats'
LABEL_KEY = 'image_class'

def transformed_name(key):
    return key + '_xf'

def make_serving_signatures(model,
                            tf_transform_output: tft.TFTransformOutput):

    model.tft_layer = tf_transform_output.transform_features_layer()

    @tf.function(input_signature=[
        tf.TensorSpec(shape=[None], dtype=tf.string, name='examples')
    ])
    def serve_tf_examples_fn(serialized_tf_example):
        raw_feature_spec = tf_transform_output.raw_feature_spec()
        raw_feature_spec.pop(LABEL_KEY)
        raw_features = tf.io.prase_example(serialized_tf_example, raw_feature_spec)
        transformed_features = model.tft_layer(raw_features)

        logging.info('serve_transformed_features = %s', transformed_features)

        output = model(transformed_features)
        return {'outputs': outputs}

    @tf.function(input_signature=[
        tf.TensorSpec(shape=[None], dtype=tf.string, name='examples')
    ])
    def transform_features_fn(serialized_tf_example):
        raw_feature_spec = tf_transform_output.raw_feature_spec()
        raw_features = tf.io.parse_example(serialized_tf_example, raw_feature_spec)
        transformed_features = model.tft_layer(raw_features)

        logging.info('eval_transformed_features = %s', transformed_features)

        return transformed_features

    return {
        'serving_default': serve_tf_examples_fn,
        'transform_features': transform_features_fn
    }

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

# Обработка входных данных
def preprocessing_fn(inputs):
    outputs = {}

    # Нормализация входных данных
    outputs[transformed_name(IMAGE_KEY)] = inputs[IMAGE_KEY] / 255.0
    outputs[transformed_name(LABEL_KEY)] = inputs[LABEL_KEY]

    return outputs
    



