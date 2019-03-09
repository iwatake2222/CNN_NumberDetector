# -*- coding: utf-8 -*-
import tensorflow as tf

# tflite_convert --output_file=conv_mnist.tflite --keras_model_file=conv_mnist.h5

if __name__ == '__main__':
	# for tensorflow 1.12
	# converter = tf.contrib.lite.TFLiteConverter.from_keras_model_file("conv_mnist.h5")
	# for tensorflow-nightly (1.14)
	converter = tf.lite.TFLiteConverter.from_keras_model_file("conv_mnist.h5")
	tflite_model = converter.convert()
	open("conv_mnist.tflite", "wb").write(tflite_model)
