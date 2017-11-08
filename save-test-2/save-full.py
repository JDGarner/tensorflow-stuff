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




# Initialize placeholders 
x = tf.placeholder(dtype = tf.float32, shape = [None, 28, 28], name="x")
y = tf.placeholder(dtype = tf.int32, shape = [None], name="y")

# Flatten the input data
images_flat = tf.contrib.layers.flatten(x)

# Fully connected layer 
logits = tf.contrib.layers.fully_connected(images_flat, 62, tf.nn.relu)

# Define a loss function
loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels = y, 
                                                                    logits = logits))
# Define an optimizer 
train_op = tf.train.AdamOptimizer(learning_rate=0.001).minimize(loss)

# Convert logits to label indexes
correct_pred = tf.argmax(logits, 1)

# Define an accuracy metric
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# print("images_flat: ", images_flat)
# print("logits: ", logits)
# print("loss: ", loss)
# print("predicted_labels: ", correct_pred)


# Load the test data
test_images, test_labels = load_data(test_data_directory)
# Transform to 28x28 and convert to greyscale
test_images28 = [transform.resize(image, (28, 28)) for image in test_images]
test_images28 = rgb2gray(np.array(test_images28))



tf.set_random_seed(1234)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver()
    for i in range(101):
        _, loss_value = sess.run([train_op, loss], feed_dict={x: images28, y: labels})
        if i % 10 == 0:
            print("EPOCH: ", i)

    # Save Model 
    saver.save(sess, 'my_test_model',global_step=1000)

    # Run predictions against the full test set.
    predicted = sess.run([correct_pred], feed_dict={x: test_images28})[0]

    # Calculate correct matches 
    match_count = sum([int(y == y_) for y, y_ in zip(test_labels, predicted)])

    # Calculate the accuracy
    print("For " + str(len(test_labels)) + " traffic signs, I predicted " + str(match_count) + " correctly :)");
