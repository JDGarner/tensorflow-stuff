
import tensorflow as tf

x = tf.placeholder("float", shape=[None, 2, 2], name="x")
y = tf.Variable([[1.0, 1.0], [0.0, 1.0]])
print tf.shape(y);
feed_dict2 = {x: [[3.0, 4.0], [5.0, 6.0]]}
matmul_operation = tf.matmul(x, y, name="op_to_restore_2")


# ham = tf.placeholder("float", name="ham")
# b1= tf.Variable(2.0,name="bias")
# feed_dict ={ham:4}
# multiply_operation = tf.multiply(ham, b1, name="op_to_restore")


sess = tf.Session()
sess.run(tf.global_variables_initializer())

saver = tf.train.Saver()
print sess.run(matmul_operation, feed_dict2)
saver.save(sess, 'my_test_model',global_step=1000)


# import tensorflow as tf

# #Prepare to feed input, i.e. feed_dict and placeholders
# sand = tf.placeholder("float", name="sand")
# # ham = tf.placeholder("float", name="ham")
# myvar= tf.Variable(2.0, name="bias")

# feed_dict ={sand:4}

# #Define a test operation that we will restore
# multiply_operation = tf.multiply(sand,myvar,name="op_to_restore")

# sess = tf.Session()
# sess.run(tf.global_variables_initializer())
# saver = tf.train.Saver()

# print sess.run(multiply_operation, feed_dict)
# saver.save(sess, 'my_test_model',global_step=1000)






# import tensorflow as tf

# x = tf.placeholder(tf.float32, shape=[2, 2])
# y = tf.constant([[1.0, 1.0], [0.0, 1.0]])
# feed_dict = {x: [[3.0, 4.0], [5.0, 6.0]]}

# z = tf.matmul(x, y, name="op_to_restore")

# sess = tf.Session()
# sess.run(tf.global_variables_initializer())
# saver = tf.train.Saver()

# print sess.run(z, feed_dict)

# saver.save(sess, 'my_test_model',global_step=1000)





