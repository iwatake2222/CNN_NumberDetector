# -*- coding: utf-8 -*-
import tensorflow as tf

# tflite_convert --output_file=mobilenetv2.tflite --keras_model_file=mobilenetv2.h5

if __name__ == '__main__':
	# for tensorflow 1.12
	# converter = tf.contrib.lite.TFLiteConverter.from_keras_model_file("mobilenetv2.h5")
	# for tensorflow-nightly (1.14)
	converter = tf.lite.TFLiteConverter.from_keras_model_file("mobilenetv2.h5")
	tflite_model = converter.convert()
	open("mobilenetv2.tflite", "wb").write(tflite_model)
