# -*- coding: utf-8 -*-
"""
Created on Sat Apr  6 15:51:57 2019

@author: tak
"""

import tensorflow as tf
from tensorflow.python.keras.models import load_model
from tensorflow.python.keras import backend as K
from tensorflow.python.framework.graph_util import convert_variables_to_constants
from tensorflow.python.keras.applications.mobilenet_v2 import MobileNetV2
from tensorflow.python.keras.applications.mobilenet_v2 import decode_predictions
from tensorflow.python.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.python.keras.preprocessing.image import load_img
import numpy as np


model = MobileNetV2()

model.summary()

img_dog = load_img('dog.jpg', target_size=(224, 224))


