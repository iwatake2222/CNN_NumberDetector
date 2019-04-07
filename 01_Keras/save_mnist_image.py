# -*- coding: utf-8 -*-


import os
from PIL import Image
from tensorflow.python.keras.datasets import mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()

base_path = 'mnist_images'
train_path = os.path.join(base_path, 'train')
test_path = os.path.join(base_path, 'test')
os.makedirs(train_path, exist_ok=True)
os.makedirs(test_path, exist_ok=True)

for i in range(10):
	os.makedirs(os.path.join(train_path, 'num_' + str(i)), exist_ok=True)
	os.makedirs(os.path.join(test_path, 'num_' + str(i)), exist_ok=True)

for i in range(y_train.shape[0]):
	label = y_train[i]
	img = Image.fromarray(x_train[i])
	img.save(os.path.join(os.path.join(train_path, 'num_' + str(label)), str(label) + '_{:08d}'.format(i) + '.png'))

for i in range(y_test.shape[0]):
	label = y_test[i]
	img = Image.fromarray(x_test[i])
	img.save(os.path.join(os.path.join(test_path, 'num_' + str(label)), str(label) + '_{:08d}'.format(i) + '.png'))

