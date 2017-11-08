# import tensorflow as tf

# w1 = tf.Variable(tf.random_normal(shape=[2]), name='w1')
# w2 = tf.Variable(tf.random_normal(shape=[5]), name='w2')

# # Create a saver.
# saver = tf.train.Saver()
# sess = tf.Session()
# for step in xrange(10000):
#     sess.run(tf.global_variables_initializer())
#     if step % 1000 == 0:
#         # Append the step number to the checkpoint name:
#         saver.save(sess, './save-test/my_test_model', global_step=step)


import tensorflow as tf
w1 = tf.Variable(tf.random_normal(shape=[9]), name='w1')
w2 = tf.Variable(tf.random_normal(shape=[5]), name='w2')
saver = tf.train.Saver()
sess = tf.Session()
sess.run(tf.global_variables_initializer())
saver.save(sess, 'my_test_model')