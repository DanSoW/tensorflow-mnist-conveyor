import tensorflow as tf
import tensorflow_transform as tft

from tfx.components.trainer.fn_args_utils import FnArgs
import mnist_utils_native_keras_base as base


# Получение функции для парсинга сериализованных данных tf.Example
def _get_serve_tf_examples_fn(model, tf_transform_output):
    model.tft_layer = tf_transform_output.transform_features_layer()

    print("tf_transform_output: ", tf_transform_output)
    print("features: ", tf_transform_output.transform_features_layer())
    print("feature_spec: ", tf_transform_output.raw_feature_spec())

    @tf.function
    def serve_tf_examples_fn(serialized_tf_examples):
        feature_spec = tf_transform_output.raw_feature_spec()
        feature_spec.pop(base.LABEL_KEY)
        parsed_features = tf.io.parse_example(serialized_tf_examples, feature_spec)
        transformed_features = model.tft_layer(parsed_features)

        return {'outputs': model(transformed_features)}

    return serve_tf_examples_fn

# TFX Transform будет вызывать эту функцию
def preprocessing_fn(inputs):
    return base.preprocessing_fn(inputs)


# TFX Trainer будет вызывать эту функцию
def run_fn(fn_args: FnArgs):
    # Определяем размер пакета
    batch_size = 32

    tf_transform_output = tft.TFTransformOutput(fn_args.transform_output)
    
    # Определяем набор данных для обучения и валидации
    train_dataset = base.input_fn(fn_args.train_files, fn_args.data_accessor,
                                  tf_transform_output, batch_size)
    eval_dataset = base.input_fn(fn_args.eval_files, fn_args.data_accessor,
                                 tf_transform_output, batch_size)

    # Определяем стратегию распределённого обучения
    mirrored_strategy = tf.distribute.MirroredStrategy()

    # Генерируем модель в контексте выбранной стратегии
    with mirrored_strategy.scope():
        model = base.build_keras_model()

    # Пишем логи по пути для tensorboard
    tensorboard_callback = tf.keras.callbacks.TensorBoard(
            log_dir=fn_args.model_run_dir, update_freq='epoch')

    print("TensorBoard logs write to: ", fn_args.model_run_dir)
       
    #print(train_dataset)
    #for raw_record in train_dataset.take(1):
        #print(raw_record)
        #print(raw_record[0]["image_floats_xf"].numpy())
        #print(raw_record[1].numpy())

    #serving_model_dir = "/".join(list(fn_args.serving_model_dir.split('/')[0:-1]))
    #serving_model_dir = fn_args.serving_model_dir

    # Запускаем процесс обучения модели
    model.fit(
            train_dataset,
            epochs=32
            batch_size=batch_size,
            steps_per_epoch=fn_args.train_steps // batch_size,
            validation_data=eval_dataset,
            validation_steps=fn_args.eval_steps // batch_size,
            callbacks=[tensorboard_callback])

    #signatures = {
    #        'serving_default': 
    #            _get_serve_tf_examples_fn(
    #                model, tf_transform_output).get_concrete_function(
    #                    tf.TensorSpec(shape=[None], dtype=tf.string, name='examples')
    #                )
    #}

    # Экспорт модели
    model.export(fn_args.serving_model_dir)
    #tf.saved_model.save(model, fn_args.serving_model_dir, signatures=signatures)


