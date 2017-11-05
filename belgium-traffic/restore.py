import os
import tensorflow as tf

ROOT_PATH = "/Users/jamie/Documents/Work/python-stuff/belgium-traffic/"

with tf.Session() as sess:
  new_saver = tf.train.import_meta_graph('my_model.meta')
  new_saver.restore(sess, tf.train.latest_checkpoint('./'))
