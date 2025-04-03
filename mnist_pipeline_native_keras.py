
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

_dir_root = os.path.join('home', 'tensorflow-mnist-conveyor')

# Директория для MNIST данных
_data_root = os.path.join(_dir_root, 'data')

# Модуль для выполнения определённых пользовательских функций Transform и Trainer
_module_file = os.path.join(_dir_root, 'mnist_utils_native_keras.py')

# Путь, который будет прослушиваться сервером моделей. Pusher выведет сюда обученную модель
_serving_model_dir = os.path.join(_dir_root, 'serving_model', _pipeline_name)

_tfx_root = os.path.join('home', 'tfx')
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



