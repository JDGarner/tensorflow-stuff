import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
from sklearn.metrics import confusion_matrix

# LOAD DATA
from tensorflow.examples.tutorials.mnist import input_data
data = input_data.read_data_sets("data/MNIST/", one_hot=True)

# GET INDEXES OF EACH TEST ARRAY (e.g. [0,0,0,0,1,0] -> 4)
data.test.cls = np.array([label.argmax() for label in data.test.labels])

# MNIST images are 28*28 pixels
img_size = 28
img_size_flat = img_size * img_size
img_shape = (img_size, img_size)

num_classes = 10


# HELPER FUNCTION TO PLOT IMAGES
def plot_images(images, cls_true, cls_pred=None):
    assert len(images) == len(cls_true) == 9
    
    # Create figure with 3x3 sub-plots.
    fig, axes = plt.subplots(3, 3)
    fig.subplots_adjust(hspace=0.3, wspace=0.3)

    for i, ax in enumerate(axes.flat):
        # Plot image.
        ax.imshow(images[i].reshape(img_shape), cmap='binary')

        # Show true and predicted classes.
        if cls_pred is None:
            xlabel = "True: {0}".format(cls_true[i])
        else:
            xlabel = "True: {0}, Pred: {1}".format(cls_true[i], cls_pred[i])

        ax.set_xlabel(xlabel)
        
        # Remove ticks from the plot.
        ax.set_xticks([])
        ax.set_yticks([])


# Get the true classes for first 10 images, plot them
images = data.test.images[0:9]
cls_true = data.test.cls[0:9]
plot_images(images=images, cls_true=cls_true)

# Placeholder for images (None = arbitrary number of images) e.g. [image_index, [...image_pixel_values...]]
images_placeholder = tf.placeholder(tf.float32, [None, img_size_flat])

# What labels each image actually has e.g. [image_index, [0,0,0,1,0,0,0]]
true_image_label_placeholder = tf.placeholder(tf.float32, [None, num_classes])

# The indexes of the true class of the images e.g.  [7,4,3,6,1,5] (the index each item in this array is the image_index)
true_image_class_placeholder = tf.placeholder(tf.int64, [None])

# The weights in this layer are from each pixel value to each class, so 784*10 in total
# Starting value is 0 (could be this or random, don't think it matters)
weights = tf.Variable(tf.zeros([img_size_flat, num_classes]))

# bias values for each class
biases = tf.Variable(tf.zeros([num_classes]))

# Essentially this is the result of multiplying all the pixel values of each image with 
# weights for each class, then combining these into one value
# For each class there are 784 pixels * weights -> final value
# End result is [num_images, num_classes], so for each image there is an array of size
# 10 that shows the probability of each class
logits = tf.matmul(images_placeholder, weights) + biases

# This normalises the proabilities to a value between 0 and 1
image_predictions = tf.nn.softmax(logits)

# The most likely class that the network predicted
image_prediction_class = tf.argmax(image_predictions, dimension=1)

