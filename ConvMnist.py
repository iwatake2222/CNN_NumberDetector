# -*- coding: utf-8 -*-
from tensorflow.python.keras.datasets import mnist
from tensorflow.python.keras.utils import to_categorical
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.layers import Input, Conv2D, MaxPooling2D, Dropout, Flatten, Dense
from tensorflow.python.keras.callbacks import TensorBoard
from tensorflow.python.keras.models import load_model
import numpy as np

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
		(x_train, y_train), (x_test, y_test) = mnist.load_data()
		
		# 学習データの前処理
		# X: 6000x28x28x1のTensorに変換し、値を0～1.0に正規化
		# Y: one-hot化(6000x1 -> 6000x10)
		x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], x_train.shape[2], 1)
		x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], x_test.shape[2], 1)
		x_train = x_train / 255.
		x_test = x_test / 255.
		y_train = to_categorical(y_train, 10)
		y_test = to_categorical(y_test, 10)

		# 学習状態は悪用のTensorBoard設定
		tsb = TensorBoard(log_dir='./logs')
		
		# Convolutionモデルの作成
		input = Input(shape=(28,28,1))
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
			batch_size=128,
			epochs=10,
			validation_split=0.2,
#			callbacks=[tsb],
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
		if input_image.shape != (28,28):
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
