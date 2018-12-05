from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Imports
import numpy as numpy
import tensorflow as tf 

tf.logging.set_vervosity(tf.logging.INFO)

def cnn_model_fc(features, label, mode):
	# Input Layer
	# [batch_size, image_height, image_width, channels]
	# batch_size: -1, dynamic set up batch_size according to input
	input_layer = tf.reshape(features["x"], [-1, 28, 28, 1])

	# Convolutional Layer #1
	# filter=32*5*5
	conv1 = tf.layers.conv2d(
		inputs=input_layer,
		filter=32,
		kernel_size=[5,5],
		padding="same",		# output_layer has same size as input_layer
		activation=tf.nn.relu
		)

	# Pooling Layer #1
	pool1 = tf.layers.max_pooling2d(
		inputs=conv1,
		pool_size=[2, 2],
		strides=2
		)

	# Convolutional Layer #2
	conv2 = tf.layers.conv2d(
		inputs=pool1,
		filters=64,
		kernel_size=[5, 5],
		padding="same",
		activation=tf.nn.relu
		)

	# Pooling Layer #2
	

if __name__ == "__main__":
	tf.app.run()


