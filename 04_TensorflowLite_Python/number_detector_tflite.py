# -*- coding: utf-8 -*-
import cv2
import tensorflow as tf
import numpy as np

if __name__ == '__main__':
	# prepara input image
	img = cv2.imread('resource/4.jpg')
	cv2.imshow('image', img)
	img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	img = cv2.resize(img, (28, 28))
	img = 255 - img
	img = img.reshape(1, img.shape[0], img.shape[1], 1)	# (1, 28, 28, 1)
	img = img / 255.
	img = img.astype(np.float32)

	# load model
	# for tensorflow 1.12
	# interpreter = tf.contrib.lite.Interpreter(model_path="conv_mnist.tflite")
	# for tensorflow-nightly (1.14)
	interpreter = tf.lite.Interpreter(model_path="conv_mnist.tflite")
	interpreter.allocate_tensors()
	input_details = interpreter.get_input_details()
	output_details = interpreter.get_output_details()

	# set input tensor
	interpreter.set_tensor(input_details[0]['index'], img)

	# run
	interpreter.invoke()

	# get outpu tensor
	probs = interpreter.get_tensor(output_details[0]['index'])

	# print result
	result = np.argmax(probs[0])
	score = probs[0][result]
	print("predicted number is {} [{:.2f}]".format(result, score))

	cv2.waitKey(0)
	cv2.destroyAllWindows()

