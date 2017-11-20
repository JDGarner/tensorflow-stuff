import tensorflow as tf
# import os

# ROOT_PATH = "/Users/jamieg/Documents/Work/Hackday/tensorflow-stuff/save-test/"

# with tf.Session() as sess:
#     saver = tf.train.import_meta_graph(os.path.join(ROOT_PATH, "my_test_model-1000.meta"))
#     saver.restore(sess, ROOT_PATH)
#     print(sess.run('w1:0'))


saver = tf.train.Saver()
save_dir = 'checkpoints/'

# If the save directory doesn't exist, make it
if not os.path.exists(save_dir):
    os.makedirs(save_dir)


save_path = os.path.join(save_dir, 'best_validation')

with tf.Session() as sess:
    saver = tf.train.import_meta_graph('my_test_model.meta')
    saver.restore(sess,tf.train.latest_checkpoint('./'))
    print(sess.run('w1:0'))
    print(sess.run('w2:0'))