import tensorflow as tf
# import os

# ROOT_PATH = "/Users/jamieg/Documents/Work/Hackday/tensorflow-stuff/save-test/"

# with tf.Session() as sess:
#     saver = tf.train.import_meta_graph(os.path.join(ROOT_PATH, "my_test_model-1000.meta"))
#     saver.restore(sess, ROOT_PATH)
#     print(sess.run('w1:0'))


with tf.Session() as sess:
    saver = tf.train.import_meta_graph('my_test_model.meta')
    saver.restore(sess,tf.train.latest_checkpoint('./'))
    print(sess.run('w1:0'))
    print(sess.run('w2:0'))