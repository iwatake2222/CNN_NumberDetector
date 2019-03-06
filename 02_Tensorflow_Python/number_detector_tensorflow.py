# -*- coding: utf-8 -*-
import cv2
import tensorflow as tf
import numpy as np

if __name__ == '__main__':
	img = cv2.imread('resource/4.jpg')
	cv2.imshow('image', img)

	img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	img = cv2.resize(img, (28, 28))
	img = 255 - img
	img = img.reshape(1, img.shape[0], img.shape[1], 1)
	img = img / 255.

	with tf.Session() as sess:
		with tf.gfile.GFile('conv_mnist.pb', 'rb') as f:
			graph_def = tf.GraphDef()
			graph_def.ParseFromString(f.read())
			sess.graph.as_default()
			_ = tf.import_graph_def(graph_def)
			tensor_input = sess.graph.get_tensor_by_name('import/input_1:0')
			tensor_output = sess.graph.get_tensor_by_name('import/dense_1/Softmax:0')
			probs = sess.run(tensor_output, {tensor_input: img})

	result = np.argmax(probs[0])
	score = probs[0][result]

	print("predicted number is {} [{:.2f}]".format(result, score))

	cv2.waitKey(0)
	cv2.destroyAllWindows()

