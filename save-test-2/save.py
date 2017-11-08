import os
import skimage
import numpy as np
from skimage import data
from skimage import transform
from skimage.color import rgb2gray
import matplotlib.pyplot as plt
import tensorflow as tf

def load_data(data_directory):
    # get all the directories within data_directory
    directories = [d for d in os.listdir(data_directory) 
                   if os.path.isdir(os.path.join(data_directory, d))]

    labels = []
    images = []
    for d in directories:
        label_directory = os.path.join(data_directory, d)

        # get all the files within the directory that end with .ppm
        file_names = [os.path.join(label_directory, f) 
                      for f in os.listdir(label_directory) 
                      if f.endswith(".ppm")]
        
        # read the image in and set the label equal to the directory name
        for f in file_names:
            images.append(skimage.data.imread(f))
            labels.append(int(d))
    return images, labels

ROOT_PATH = "/Users/jamieg/Documents/Work/Hackday/tensorflow-stuff/belgium-traffic/"
train_data_directory = os.path.join(ROOT_PATH, "Training")
test_data_directory = os.path.join(ROOT_PATH, "Testing")

images, labels = load_data(train_data_directory)

# Rescale images
images28 = [transform.resize(image, (28, 28)) for image in images]
# Convert to grayscale
images28 = np.array(images28)
images28 = rgb2gray(images28)

# import tensorflow as tf


x = tf.placeholder("float", shape=[None, 2, 2], name="x")
y = tf.Variable([[1.0, 1.0], [0.0, 1.0]])
feed_dict2 = {x: images28}
# reverse_operation = tf.reverse(x, y, name="op_to_restore_2")

# ham = tf.placeholder("float", name="ham")
# b1= tf.Variable(2.0,name="bias")
# feed_dict ={ham:4}
# multiply_operation = tf.multiply(ham, b1, name="op_to_restore")

sess = tf.Session()
sess.run(tf.global_variables_initializer())

saver = tf.train.Saver()
print sess.run(x, feed_dict2)
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





