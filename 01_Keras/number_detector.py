# -*- coding: utf-8 -*-
import cv2
from ConvMnist import ConvMnist

if __name__ == '__main__':
	conv_mnist = ConvMnist(filename='conv_mnist.h5')
	
	img = cv2.imread('resource/4.jpg')
	cv2.imshow('image', img)

	img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	img = cv2.resize(img, (28, 28))
	img = 255 - img
	
	result, score = conv_mnist.predict(img)
	print("predicted number is {} [{:.2f}]".format(result, score))

	cv2.waitKey(0)
	cv2.destroyAllWindows()

