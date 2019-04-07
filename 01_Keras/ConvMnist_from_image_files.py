# -*- coding: utf-8 -*-
import os
import math
import pickle
import numpy as np
from tensorflow.python.keras.models import load_model
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.layers import Input, Conv2D, MaxPooling2D, Dropout, Flatten, Dense
from tensorflow.python.keras.preprocessing.image import load_img
from tensorflow.python.keras.preprocessing.image import img_to_array, array_to_img
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.keras.callbacks import ModelCheckpoint

MODEL_WIDTH = 28
MODEL_HEIGHT = 28
BATCH_SIZE = 64
EPOCH_NUM = 5

class ConvMnist:

	def __init__(self, filename=None):
		'''
		学習済みモデルファイルをロードする (optional)
		'''
		self.model = None
		if filename is not None:
			print('load model: ', filename)
			self.model = load_model(filename)
			self.model.summary()

	def train(self):
		'''
		学習する
		'''
		# MNISTの学習用データ、テストデータをロードする
		(x_train, y_train), (x_test, y_test) = self.load_data()
		
		# Convolutionモデルの作成
		input = Input(shape=(MODEL_WIDTH,MODEL_HEIGHT,1))
		conv1 = Conv2D(
					filters=8,
					kernel_size=(3,3),
					strides=(1,1),
					padding='same',
					activation='relu'
				)(input)
		pool1 = MaxPooling2D(pool_size=(2,2))(conv1)
		conv2 = Conv2D(
					filters=4,
					kernel_size=(3,3),
					strides=(1,1),
					padding='same',
					activation='relu'
				)(pool1)
		dropout1 = Dropout(0.2)(conv2)
		flatten1 = Flatten()(dropout1)
		output = Dense(units=10, activation='softmax')(flatten1)
		self.model = Model(inputs=[input], outputs=[output])
		
		self.model.summary()
		
		self.model.compile(
			optimizer='adam',
			loss='categorical_crossentropy',
			metrics=['accuracy']
		)
		
		# Convolutionモデルの学習
		self.model.fit(
				x_train,
				y_train,
				batch_size=BATCH_SIZE,
				epochs=EPOCH_NUM,
				validation_split=0.2,
		)
		
		# 学習したモデルを使用して、テスト用データで評価する
		score = self.model.evaluate(x_test, y_test, verbose=0)
		print("test data score: ", score)

	def save_trained_model(self, filename):
		'''
		学習済みモデルをファイル(h5)に保存する
		'''
		self.model.save(filename)


	def predict(self, input_image):
		'''
		1つのグレースケール入力画像(28x28のndarray)に対して、数字(0～9)を判定する
		ret: result, score
		'''
		if input_image.shape != (MODEL_WIDTH, MODEL_HEIGHT):
			return -1, -1
		input_image = input_image.reshape(1, input_image.shape[0], input_image.shape[1], 1)
		input_image = input_image / 255.

		probs = self.model.predict(input_image)
		result = np.argmax(probs[0])
		return result, probs[0][result]

	def load_data(self):
		filename = 'mnist_train_data.pickle'
		if os.path.exists(filename):
			with open(filename,'rb') as f:
				data = pickle.load(f)
			x_train = data[0][0]
			y_train = data[0][1]
			x_test = data[1][0]
			y_test = data[1][1]
			x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], x_train.shape[2], 1)
			x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], x_test.shape[2], 1)
		else:
			idg_train = ImageDataGenerator(rescale=1/255.)
	
			# training用データのgenerator
			img_itr_train = idg_train.flow_from_directory(
					'mnist_images/train',
					color_mode="grayscale",
					target_size=(MODEL_WIDTH, MODEL_HEIGHT),
					batch_size=1,
					class_mode='categorical',
			)
			
			img_itr_train.batch_size = img_itr_train.samples
			x_train, y_train = next(img_itr_train)
			
			# test用データのgenerator
			img_itr_test = idg_train.flow_from_directory(
					'mnist_images/test',
					color_mode="grayscale",
					target_size=(MODEL_WIDTH, MODEL_HEIGHT),
					batch_size=1,
					class_mode='categorical',
			)
			
			img_itr_test.batch_size = img_itr_test.samples
			x_test, y_test = next(img_itr_test)
			
			with open(filename,'wb') as f:
				pickle.dump(((x_train, y_train), (x_test, y_test)), f)

		return (x_train, y_train), (x_test, y_test)


if __name__ == '__main__':
	conv_mnist = ConvMnist()
	conv_mnist.train()
	conv_mnist.save_trained_model(filename='conv_mnist.h5')

	'''sample code to predict'''
#	conv_mnist = ConvMnist(filename='conv_mnist.h5')
#	result = conv_mnist.predict(conv_mnist.x_test[0])
#	print("answer = {}, predict = {}" .format(conv_mnist.y_test[0], result))

