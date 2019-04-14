# -*- coding: utf-8 -*-
"""
Created on Sat Apr  6 15:51:57 2019

@author: tak
"""
import os
import math
import pickle
from PIL import Image
import numpy as np
from tensorflow.python.keras.datasets import mnist
from tensorflow.python.keras.utils import to_categorical
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.models import load_model
from tensorflow.python.keras.layers import Dense, Dropout, Flatten, Input, Conv2D, MaxPooling2D
from tensorflow.python.keras.preprocessing.image import load_img
from tensorflow.python.keras.preprocessing.image import img_to_array, array_to_img
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.keras.callbacks import ModelCheckpoint
from tensorflow.python.keras.callbacks import TensorBoard
from tensorflow.python.keras.applications.mobilenet_v2 import MobileNetV2
from tensorflow.python.keras.applications.mobilenet_v2 import decode_predictions
from tensorflow.python.keras.applications.mobilenet_v2 import preprocess_input

'''
モデル用パラメータ
'''
MODEL_WIDTH = 128
MODEL_HEIGHT = 128
MODEL_CHANNEL = 3
BATCH_SIZE = 32
EPOCH_NUM = 30


def load_data():
	'''
	MNISTの学習用データ、テストデータをロードする
	'''
	filename = 'mnist_train_data.pickle'
	
	if os.path.exists(filename):
		# 2回目以降はpickleファイルからロードする
		with open(filename,'rb') as f:
			data = pickle.load(f)
		x_train = data[0][0]
		y_train = data[0][1]
		x_test = data[1][0]
		y_test = data[1][1]
		x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], x_train.shape[2], MODEL_CHANNEL)
		x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], x_test.shape[2], MODEL_CHANNEL)
	else:
		# 初回はデータを作成して、pickleファイルに保存する
		(x_train_org, y_train), (x_test_org, y_test) = mnist.load_data()
		
		# limit the number of images to reduce resource usage
#		x_train_org = x_train_org[:20000, :, :]
#		y_train = y_train[:20000]
#		x_test_org = x_test_org[:5000, :, :]
#		y_test = y_test[:5000]
		
		# Change Image format
		x_train = np.empty((y_train.shape[0], MODEL_WIDTH, MODEL_HEIGHT, MODEL_CHANNEL), dtype=np.uint8)
		for i in range(y_train.shape[0]):
			img = Image.fromarray(x_train_org[i])
			img = img.convert('RGB') if MODEL_CHANNEL == 3 else img.convert('L')
			img = img.resize((MODEL_WIDTH, MODEL_HEIGHT))
			x_train[i] = img_to_array(img)
		
		x_test = np.empty((y_test.shape[0], MODEL_WIDTH, MODEL_HEIGHT, MODEL_CHANNEL), dtype=np.uint8)
		for i in range(y_test.shape[0]):
			img = Image.fromarray(x_test_org[i])
			img = img.convert('RGB') if MODEL_CHANNEL == 3 else img.convert('L')
			img = img.resize((MODEL_WIDTH, MODEL_HEIGHT))
			x_test[i] = img_to_array(img)
		
		# 保存
		pickle.dump(((x_train, y_train), (x_test, y_test)), open(filename,'wb'), protocol=4)
	
	return (x_train, y_train), (x_test, y_test)

'''
学習/テスト用データ読み込み
'''
try:
	x_train
except NameError:
	(x_train, y_train), (x_test, y_test) = load_data()
	
	# 前処理
	x_train = preprocess_input(x_train)
	x_test = preprocess_input(x_test)
	y_train = to_categorical(y_train, 10)
	y_test = to_categorical(y_test, 10)

'''
モデル読み込み、出力層を付けて再構築
'''
model_base = MobileNetV2(include_top=False, weights='imagenet', input_shape=(MODEL_WIDTH, MODEL_HEIGHT, MODEL_CHANNEL))

# 出力層を変えるだけだと、精度が出なかった
#for layer in model_base.layers:
#	layer.trainable = False
#for layer in model_base.layers[:int(len(model_base.layers) / 2)]:
#	layer.trainable = False
#model_base = MobileNetV2(include_top=False, weights=None, input_shape=(MODEL_WIDTH, MODEL_HEIGHT, MODEL_CHANNEL))

output_without_fc = model_base.output
flatten1 = Flatten()(output_without_fc)
fc1 = Dense(units=64, activation='sigmoid')(flatten1)
dropout1 = Dropout(0.2)(fc1)
fc2 = Dense(units=10, activation='softmax')(dropout1)
model = Model(inputs=model_base.input, outputs=fc2)

model.compile(
	optimizer='adam',
	loss='categorical_crossentropy',
	metrics=['accuracy'],
)

model.summary()

'''
学習する
'''
cp = ModelCheckpoint(
		'ep_{epoch:02d}_val_acc_{val_acc:.2f}.h5',
		monitor='val_acc',
		period=10,
		)
model.fit(
		x_train,
		y_train,
		batch_size=BATCH_SIZE,
		epochs=EPOCH_NUM,
		validation_split=0.2,
		callbacks = [TensorBoard(log_dir='./logs'), cp]
)


'''
テスト用データで評価する
'''
score = model.evaluate(x_test, y_test, verbose=0)
print("test data score: ", score)


# 学習したモデルを保存
model.save('mobilenet_v2_mnist.h5')




import cv2

img = cv2.imread('resource/4.jpg')
img = cv2.resize(img, (128, 128))
img = 255 - img

img = img.reshape(1, img.shape[0], img.shape[1], 3)
img = preprocess_input(img)

print('4')
probs = model.predict(img)
result = np.argmax(probs[0])
print("predicted number is {} [{:.2f}]".format(result, probs[0][result]))

print(y_train[10])
probs = model.predict(x_train[10].reshape(1, 128, 128,3))
result = np.argmax(probs[0])
print("predicted number is {} [{:.2f}]".format(result, probs[0][result]))


