import time
print("fff")
import models
import numpy as np
import partition_utils
print("ggg")
import tensorflow.compat.v1 as tf

tf.disable_eager_execution()
tf.disable_v2_behavior()
print(tf.__version__)
print(tf.test.is_built_with_cuda())  # Check if TensorFlow is built with CUDA
print(tf.config.list_physical_devices('GPU'))  # Check if GPUs are visible

import utils
from tensorflow.python.client import device_lib

import time
