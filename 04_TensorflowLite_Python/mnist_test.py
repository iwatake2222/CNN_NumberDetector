# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np
from tensorflow.python.keras.datasets import mnist

if __name__ == '__main__':
	# prepara input images
	(x_train, y_train), (x_test, y_test) = mnist.load_data()
	x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], x_test.shape[2], 1)
	x_test = x_test / 255.
	x_test = x_test.astype(np.float32)

	## load model
	# for tensorflow 1.12
	# interpreter = tf.contrib.lite.Interpreter(model_path="conv_mnist.tflite")
	# for tensorflow-nightly (1.14)
	interpreter = tf.lite.Interpreter(model_path="conv_mnist.tflite")
	interpreter.allocate_tensors()
	input_details = interpreter.get_input_details()
	output_details = interpreter.get_output_details()

	results = []
	for i in range(x_test.shape[0]):
		# set input tensor
		x = x_test[i].reshape(1,28,28,1)
		interpreter.set_tensor(input_details[0]['index'], x)
		
		# run
		interpreter.invoke()

		# get outpu tensor
		probs = interpreter.get_tensor(output_details[0]['index'])
		results.append(np.argmax(probs))

	# print result
	print("accuracy = ", 1.0 * np.sum(y_test == results) / y_test.size)
