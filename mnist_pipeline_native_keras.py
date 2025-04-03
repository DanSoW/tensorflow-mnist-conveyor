
import os
from typing import List

import absl
import tensorflow_model_analysis as tfma
from tfx.components import Evaluator
from tfx.components import ExampleValidator
from tfx.components import ImportExampleGen
from tfx.components import Pusher
from tfx.components import SchemaGen
from tfx.components import StatisticsGen
from tfx.components import Trainer
from tfx.components import Transform
from tfx.orchestration import metadata
from tfx.orchestration import pipeline
from tfx.orchestration.beam.beam_dag_runner import BeamDagRunner
from tfx.proto import pusher_pb2
from tfx.proto import trainer_pb2

_pipeline_name = 'mnist_native_keras'

_dir_root = os.path.join('/home', 'tensorflow-mnist-conveyor')

# Директория для MNIST данных
_data_root = os.path.join(_dir_root, 'data')

# Модуль для выполнения определённых пользовательских функций Transform и Trainer
_module_file = os.path.join(_dir_root, 'mnist_utils_native_keras.py')

# Путь, который будет прослушиваться сервером моделей. Pusher выведет сюда обученную модель
_serving_model_dir = os.path.join(_dir_root, 'serving_model', _pipeline_name)

_tfx_root = os.path.join('/home', 'tfx')
_pipeline_root = os.path.join(_tfx_root, 'pipelines', _pipeline_name)

# Путь до хранения ML-метаданных SQLite
_metadata_path = os.path.join(_tfx_root, 'metadata', _pipeline_name, 'metadata.db')

# Аргументы конвейера для Beam
_beam_pipeline_args = [
        '--direct_running_mode=multi_processing',
        # 0 означает автоматическое определение в зависимости от количества
        # доступных процессоров во время выполнения
        '--direct_num_workers=0',
]

print(_dir_root)
print(_data_root)
print(_module_file)
print(_serving_model_dir)
print(_tfx_root)
print(_pipeline_root)
print(_metadata_path)

# Создание пайплайна с определёнными параметрами для классификации рукописных цифр MNIST
def _create_pipeline(pipeline_name: str, pipeline_root: str, data_root: str,
                     module_file: str, serving_model_dir: str, metadata_path: str,
                     beam_pipeline_args: List[str], accuracy_threshold: float = 0.8) -> pipeline.Pipeline:
    
    # Импорт данных в конвейер
    example_gen = ImportExampleGen(input_base=data_root)

    # Вычисляет статистику по данным для визуализации и проверки на примере
    statistics_gen = StatisticsGen(examples=example_gen.outputs['examples'])

    # Генерация схема на основе файлов статистики
    schema_gen = SchemaGen(
            statistics=statistics_gen.outputs['statistics'], infer_feature_shape=True)

    # Выполняет обнаружение аномалий на основе статистики и схемы данных
    example_validator = ExampleValidator(
            statistics=statistics_gen.outputs['statistics'],
            schema=schema_gen.outputs['schema'])

    # Преобразование данных
    transform = Transform(
            examples=example_gen.outputs['examples'],
            schema=schema_gen.outputs['schema'],
            module_file=module_file)
    
    # Создание компонента Trainer
    def _create_trainer(module_file, component_id):
        return Trainer(
                module_file=module_file,
                examples=transform.outputs['transformed_examples'],
                transform_graph=transform.outputs['transform_graph'],
                schema=schema_gen.outputs['schema'],
                train_args=trainer_pb2.TrainArgs(num_steps=5000),
                eval_args=trainer_pb2.EvalArgs(num_steps=100)).with_id(component_id)

    trainer = _create_trainer(module_file, 'Trainer.mnist')
    
    # Конфигурация для оценки качества модели-кандидата
    eval_config = tfma.EvalConfig(
            model_specs=[tfma.ModelSpec(label_key='image_class')],
            slicing_specs=[tfma.SlicingSpec()],
            metrics_specs=[
                tfma.MetricsSpec(metrics=[
                    tfma.MetricConfig(
                        class_name='SparseCategoricalAccuracy',
                        threshold=tfma.MetricThreshold(
                            value_threshold=tfma.GenericValueThreshold(
                                lower_bound={'value': accuracy_threshold})))
                ])
            ])

    # Использует TFMA для вычисления статистики оценки характеристик модели.
    evaluator = Evaluator(
            examples=example_gen.outputs['examples'],
            model=trainer.outputs['model'],
            eval_config=eval_config).with_id('Evaluator.mnist')

    # Проверяет, прошла ли модель этапы проверки, и отправляет модель
    # в пункт назначения файла, если проверка пройдена.
    pusher = Pusher(
            model=trainer.outputs['model'],
            model_blessing=evaluator.outputs['blessing'],
            push_destination=pusher_pb2.PushDestination(
                filesystem=pusher_pb2.PushDestination.Filesystem(
                    base_directory=serving_model_dir))).with_id('Pusher.mnist')

    
    return pipeline.Pipeline(
            pipeline_name=pipeline_name,
            pipeline_root=pipeline_root,
            components=[
                example_gen,
                statistics_gen,
                schema_gen,
                example_validator,
                transform,
                trainer,
                evaluator,
                pusher,
            ],
            enable_cache=True,
            metadata_connection_config=metadata.sqlite_metadata_connection_config(
                metadata_path),
            beam_pipeline_args=beam_pipeline_args)


if __name__ == '__main__':
    absl.logging.set_verbosity(absl.logging.INFO)
    BeamDagRunner().run(
            _create_pipeline(
                pipeline_name=_pipeline_name,
                pipeline_root=_pipeline_root,
                data_root=_data_root,
                module_file=_module_file,
                serving_model_dir=_serving_model_dir,
                metadata_path=_metadata_path,
                beam_pipeline_args=_beam_pipeline_args))


    





