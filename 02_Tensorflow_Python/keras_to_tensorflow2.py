# -*- coding: utf-8 -*-
import tensorflow as tf
from tensorflow.python.keras.models import load_model

input_keras_model = './conv_mnist.h5'
export_dir = './conv_mnist_pb'

if __name__ == '__main__':
	sess = tf.Session()
	sess.run(tf.global_variables_initializer())
	tf.keras.backend.set_session(sess)
	model = load_model(input_keras_model)
	builder = tf.saved_model.builder.SavedModelBuilder(export_dir)
	signature = tf.saved_model.predict_signature_def(inputs={t.name:t for t in model.inputs},
												  outputs={t.name:t for t in model.outputs})
	builder.add_meta_graph_and_variables(sess,
									  tags=[tf.saved_model.tag_constants.SERVING],
									  signature_def_map={'predict': signature})
	builder.save(as_text=True)
	sess.close()
	
	print('output_node_names:')
	for t in model.inputs:
		print(t.name)
	
	print('output_node_names:')
	for t in model.outputs:
		print(t.name)


# Terminalで実行(改行区切りはWindows用。Linuxでは\)
'''
freeze_graph ^
--input_saved_model_dir=./conv_mnist_pb ^
--output_graph=conv_mnist.pb ^
--output_node_names=dense/Softmax ^
--clear_devices
'''

