# -*- coding: utf-8 -*-
"""
Created on Sat Apr  6 15:51:57 2019

@author: tak
"""
from tensorflow.python.keras.applications.mobilenet_v2 import MobileNetV2
from tensorflow.python.keras.applications.mobilenet_v2 import decode_predictions
from tensorflow.python.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.python.keras.preprocessing.image import load_img
from tensorflow.python.keras.preprocessing.image import img_to_array
import numpy as np

model = MobileNetV2()
model.summary()

model.save('mobilenetv2.h5')

img_dog = load_img('dog.jpg', target_size=(224, 224))
arr_dog = preprocess_input(img_to_array(img_dog))

arr_input = np.stack([arr_dog, ])

probs = model.predict(arr_input)
results = decode_predictions(probs)

print(results[0])

