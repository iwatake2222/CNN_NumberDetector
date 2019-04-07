# -*- coding: utf-8 -*-
"""
Created on Sat Apr  6 15:51:57 2019

@author: tak
"""
import os
import math
import numpy as np
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.models import load_model
from tensorflow.python.keras.layers import Dense, Dropout, Flatten, Input, Conv2D, MaxPooling2D
from tensorflow.python.keras.preprocessing.image import load_img
from tensorflow.python.keras.preprocessing.image import img_to_array, array_to_img
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.keras.callbacks import ModelCheckpoint
from tensorflow.python.keras.applications.mobilenet_v2 import MobileNetV2
from tensorflow.python.keras.applications.mobilenet_v2 import decode_predictions
from tensorflow.python.keras.applications.mobilenet_v2 import preprocess_input

MODEL_WIDTH = 96
MODEL_HEIGHT = 96
BATCH_SIZE = 128
EPOCH_NUM = 3

#model_base = MobileNetV2(include_top=False, weights=None, input_shape=(MODEL_WIDTH, MODEL_HEIGHT, 3))
model_base = MobileNetV2(include_top=False, weights='imagenet', input_shape=(MODEL_WIDTH, MODEL_HEIGHT, 3))
for layer in model_base.layers:
	layer.trainable = False

#for layer in model_base.layers[:len(model_base.layers)-10]:
#	layer.trainable = False

output_without_fc = model_base.output
flatten1 = Flatten()(output_without_fc)
fc1 = Dense(units=256, activation='sigmoid')(flatten1)
dropout1 = Dropout(0.2)(fc1)
fc2 = Dense(units=10, activation='softmax')(dropout1)
model = Model(inputs=model_base.input, outputs=fc2)

model.compile(
	optimizer='adam',
	loss='categorical_crossentropy',
	metrics=['accuracy']
)

model.summary()

'''
学習する
'''
### 画像を一度全部メモリに格納する (ラベル付けなどを自動化するためImageDataGeneratorは使う)
# データセットをディレクトリ内画像から読み込む用意
idg_train = ImageDataGenerator(
		rescale=1/255.,
		preprocessing_function=preprocess_input
)

# training用データのgenerator
img_itr_train = idg_train.flow_from_directory(
		'mnist_images/train',
#		color_mode="grayscale",
		target_size=(MODEL_WIDTH, MODEL_HEIGHT),
		batch_size=BATCH_SIZE,
		class_mode='categorical',
)

img_itr_train.batch_size=img_itr_train.samples
x_train, y_train = next(img_itr_train)

model.fit(
		x_train,
		y_train,
		batch_size=BATCH_SIZE,
		epochs=EPOCH_NUM,
		validation_split=0.2,
)


# 学習したモデルを保存
model.save('mobilenet_v2_mnist.h5')

#model = load_model("mobilenet_v2_mnist.h5")

# テスト用データで評価する
idg_test = ImageDataGenerator(
		rescale=1/255.,
		preprocessing_function=preprocess_input
)

img_itr_test = idg_test.flow_from_directory(
		'mnist_images/test',
#		color_mode="grayscale",
		target_size=(MODEL_WIDTH, MODEL_HEIGHT),
		batch_size=BATCH_SIZE,
		class_mode=None,
		shuffle=False
)

# 識別処理実施
probs = model.predict_generator(
		img_itr_test,
		steps=math.ceil(img_itr_test.samples/BATCH_SIZE)
)

# 識別精度を計算
predictions = np.argmax(probs, axis=1)
print( 1.0 * np.sum(predictions == img_itr_test.classes) / img_itr_test.n)

