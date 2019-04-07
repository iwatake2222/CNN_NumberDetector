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

	
	def preprocess_input(x, **kwargs):
		'''
		画像前処理(ここでは何もしない)
		'''
#		x = 255 - x
		return x.astype(np.float32)

	def train(self):
		'''
		学習する
		'''
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
		

		## fit_generatorを使用する場合。windowsだと遅い(画像読み込みをうまく並列化できない)
		# データセットをディレクトリ内画像から読み込む用意
		idg_train = ImageDataGenerator(
				validation_split=0.2,
				rescale=1/255.,
		#		preprocessing_function=preprocess_input
		)
		
		# training用データのgenerator(全学習画像の内80%)
		img_itr_train = idg_train.flow_from_directory(
				'mnist_images/train',
				subset="training",
				color_mode="grayscale",
				target_size=(MODEL_WIDTH, MODEL_HEIGHT),
				batch_size=BATCH_SIZE,
				class_mode='categorical',
		)
		
		# validation用データのgenerator(全学習画像の内20%)
		img_itr_validation = idg_train.flow_from_directory(
				'mnist_images/train',
				subset="validation",
				color_mode="grayscale",
				target_size=(MODEL_WIDTH, MODEL_HEIGHT),
				batch_size=BATCH_SIZE,
				class_mode='categorical',
		)
		
		# Convolutionモデルの学習
		self.model.fit_generator(
				img_itr_train,
				steps_per_epoch=math.ceil(img_itr_train.samples/BATCH_SIZE),
				epochs=EPOCH_NUM,
				validation_data=img_itr_validation,
				validation_steps=math.ceil(img_itr_validation.samples/BATCH_SIZE),
		)
		
		# テスト用データで評価する
		idg_test = ImageDataGenerator(
				rescale=1/255.,
		#		preprocessing_function=preprocess_input
		)
		
		img_itr_test = idg_test.flow_from_directory(
				'mnist_images/test',
				color_mode="grayscale",
				target_size=(MODEL_WIDTH, MODEL_HEIGHT),
				batch_size=BATCH_SIZE,
				class_mode=None,
				shuffle=False
		)
		
		# 識別処理実施
		probs = self.model.predict_generator(
				img_itr_test,
				steps=math.ceil(img_itr_test.samples/BATCH_SIZE)
		)
		
		# 識別精度を計算
		predictions = np.argmax(probs, axis=1)
		print("score: " + str(1.0 * np.sum(predictions == img_itr_test.classes) / img_itr_test.n))


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

if __name__ == '__main__':
	conv_mnist = ConvMnist()
	conv_mnist.train()
	conv_mnist.save_trained_model(filename='conv_mnist.h5')

	'''sample code to predict'''
#	conv_mnist = ConvMnist(filename='conv_mnist.h5')
#	result = conv_mnist.predict(conv_mnist.x_test[0])
#	print("answer = {}, predict = {}" .format(conv_mnist.y_test[0], result))

