from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import argparse

import tensorflow as tf 
from tensorflow.examples.tutorials.mnist import input_data

FLAGS = None

def main():
	if tf.gfile.Exists(FLAGS.log_dir):
		rf.gfile.DeleteRecursively(FLAGS.log_dir)
	tf.gfile.MakeDirs(FLAGS.log_dir)
	train()

if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument(
		'--log_dir',
		type=str,
		default=os.path.join(os.getenv('TEST_TMPDIR', '/tmp'), 'tensorflow/mnist/logs/mnist_with_summaries'),
		help='Summaries log directory'
		)
	FLAGS, unparsed = parser.parse_known_args()
	tf.app.run(main=main, argv=[sys.argv[0]]+unparsed)
