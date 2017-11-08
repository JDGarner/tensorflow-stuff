
import tensorflow as tf

sess=tf.Session()    
#First let's load meta graph and restore weights
saver = tf.train.import_meta_graph('my_test_model-1000.meta')
saver.restore(sess,tf.train.latest_checkpoint('./'))


# Now, let's access and create placeholders variables and
# create feed-dict to feed new data

graph = tf.get_default_graph()
x = graph.get_tensor_by_name("x:0")
feed_dict = {x: [[3.0, 4.0], [5.0, 6.0]]}

#Now, access the op that you want to run. 
# op_to_restore = graph.get_tensor_by_name("op_to_restore_2:0")

print sess.run(x,feed_dict)



# import tensorflow as tf

# sess=tf.Session()
# #First let's load meta graph and restore weights
# saver = tf.train.import_meta_graph('my_test_model-1000.meta')
# saver.restore(sess,tf.train.latest_checkpoint('./'))


# # Now, let's access and create placeholders variables and
# # create feed-dict to feed new data
# graph = tf.get_default_graph()
# sand = graph.get_tensor_by_name("sand:0")
# # w2 = graph.get_tensor_by_name("w2:0")
# # x = graph.get_tensor_by_name("x:0")
# feed_dict = {sand:13}

# #Now, access the op that you want to run. 
# op_to_restore = graph.get_tensor_by_name("op_to_restore:0")

# print sess.run(op_to_restore,feed_dict)
# print(sess.run('sand:0'))
# #This will print 60 which is calculated
# #using new values of w1 and w2 and saved value of b1. 