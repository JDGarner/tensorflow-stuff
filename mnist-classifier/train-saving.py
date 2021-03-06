import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
from sklearn.metrics import confusion_matrix

import time
from datetime import timedelta
import math
import os
# Use PrettyTensor to simplify Neural Network construction.
import prettytensor as pt


# -------------------------------
# LOADING THE DATASET
# -------------------------------

from tensorflow.examples.tutorials.mnist import input_data
data = input_data.read_data_sets("data/MNIST/", one_hot=True)

# GET INDEXES OF EACH TEST ARRAY (e.g. [0,0,0,0,1,0] -> 4)
data.test.cls = np.array([label.argmax() for label in data.test.labels])

# MNIST images are 28*28 pixels
img_size = 28
img_size_flat = img_size * img_size
img_shape = (img_size, img_size)

num_classes = 10




# Helper Function to plot images
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
    plt.show(block=True)


# -------------------------------
# CHECKING THE DATA
# -------------------------------
# Get the true classes for first 10 images, plot them
# images = data.test.images[0:9]
# cls_true = data.test.cls[0:9]
# plot_images(images=images, cls_true=cls_true)






# -------------------------------
# CREATING THE NETWORK MODEL
# -------------------------------

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
image_prediction_class = tf.argmax(image_predictions, axis=1)

# To improve classifier need to change the variables for weights and biases
# First need to compare the prediction outputs to the desired output
# cross-entropy is a performance measure used in classification. 
# If prediction === desired then the cross-entropy equals zero.
# Goal of optimization is to minimize the cross-entropy by changing the weights and biases of the model.

# The tensorflow function uses logits because it also calculates the softmax internally.
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=true_image_label_placeholder)

# To use the cross-entropy to guide the optimization we need a single scalar value
# Take the average of the cross-entropy for all the image classifications.
cost = tf.reduce_mean(cross_entropy)

# Now we need an optimiser function to minimise the cost
# We use a method called Gradient Descent where the step-size is set to 0.5.
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.5).minimize(cost)

# Performance measure - an array of booleans to say whether predicted class = true class
correct_predictions = tf.equal(image_prediction_class, true_image_class_placeholder)

# Performance measure - percentage of correctly predicted classes
accuracy = tf.reduce_mean(tf.cast(correct_predictions, tf.float32))






# -------------------------------
# CREATE SAVER FOR SAVING THE NETWORK
# -------------------------------
saver = tf.train.Saver()
save_dir = 'checkpoints/'

# If the save directory doesn't exist, make it
if not os.path.exists(save_dir):
    os.makedirs(save_dir)


save_path = os.path.join(save_dir, 'best_validation')









# -------------------------------
# RUNNING TENSORFLOW
# -------------------------------
session = tf.Session()
# Initialise the variables for weights and biases
session.run(tf.global_variables_initializer())

# Number of images per batch of optimsation
batch_size = 100

# Helper function to perform a set number of optimisation iterations
def optimize(num_iterations):
    for i in range(num_iterations):
        # Get a batch of training data
        images_batch, image_labels_batch = data.train.next_batch(batch_size)
        
        # Load the batch data into our placeholders
        feed_dict_train = {
          images_placeholder: images_batch,
          true_image_label_placeholder: image_labels_batch
        }

        # Run the optimizer using this batch of training data
        session.run(optimizer, feed_dict=feed_dict_train)


feed_dict_test = {
  images_placeholder: data.test.images,
  true_image_label_placeholder: data.test.labels,
  true_image_class_placeholder: data.test.cls
}

# Function for printing accuracy of network on the test data
def print_accuracy():
    # Use TensorFlow to compute the accuracy.
    acc = session.run(accuracy, feed_dict=feed_dict_test)
    
    # Print the accuracy.
    print("Accuracy on test-set: {0:.1%}".format(acc))


# Function for printing and plotting the confusion matrix
def print_confusion_matrix():
    # Get true classes for test-set.
    true_classes = data.test.cls
    
    # Get predictions for test-set
    predicted_classes = session.run(image_prediction_class, feed_dict=feed_dict_test)

    # Get the confusion matrix and plot it
    cm = confusion_matrix(y_true=true_classes, y_pred=predicted_classes)
    print(cm)
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)

    # Make various adjustments to the plot.
    plt.tight_layout()
    plt.colorbar()
    tick_marks = np.arange(num_classes)
    plt.xticks(tick_marks, range(num_classes))
    plt.yticks(tick_marks, range(num_classes))
    plt.xlabel('Predicted')
    plt.ylabel('True')



# Function for plotting mis-classified images
def plot_example_errors():
    # correct = list of booleans for if each image has been correctly classified
    # predictions = list of predicted class of each image
    correct, predictions = session.run([correct_predictions, image_prediction_class],
                                    feed_dict=feed_dict_test)

    # Get incorrectly classified images
    incorrect = (correct == False)
    images = data.test.images[incorrect]
    
    # Get the true & predicted classes for those images.
    cls_pred = predictions[incorrect]
    cls_true = data.test.cls[incorrect]
    
    # Plot the first 9 images.
    plot_images(images=images[0:9],
                cls_true=cls_true[0:9],
                cls_pred=cls_pred[0:9])





# Function for plotting the weights of the model
# 10 images are plotted, one for each digit that the model is trained to recognize.
def plot_weights():
    # Get the values for the weights from the TensorFlow variable.
    w = session.run(weights)
    
    # Get the lowest and highest values for the weights.
    w_min = np.min(w)
    w_max = np.max(w)

    # Create figure with 3x4 sub-plots,
    # where the last 2 sub-plots are unused.
    fig, axes = plt.subplots(3, 4)
    fig.subplots_adjust(hspace=0.3, wspace=0.3)

    for i, ax in enumerate(axes.flat):
        # Only use the weights for the first 10 sub-plots.
        if i<10:
            # Get the weights for the i'th digit and reshape it.
            # Note that w.shape == (img_size_flat, 10)
            image = w[:, i].reshape(img_shape)

            # Set the label for the sub-plot.
            ax.set_xlabel("Weights: {0}".format(i))

            # Plot the image.
            ax.imshow(image, vmin=w_min, vmax=w_max, cmap='seismic')

        # Remove ticks from each sub-plot.
        ax.set_xticks([])
        ax.set_yticks([])
    plt.show()



optimize(num_iterations=990)
print_accuracy()
# plot_example_errors()
# plot_weights()








