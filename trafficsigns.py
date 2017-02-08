from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import os
import skimage.data
from skimage.color import rgb2gray
import skimage.transform
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from Dataset import DataSet
import time

import tensorflow as tf
sess = tf.InteractiveSession()

#todo: move function to a separate file
def load_data(data_dir):
    """Loads a data set and returns two lists:
    
    images: a list of Numpy arrays, each representing an image.
    labels: a list of numbers that represent the images labels.
    """
    # Get all subdirectories of data_dir. Each represents a label.
    directories = [d for d in os.listdir(data_dir) 
                   if os.path.isdir(os.path.join(data_dir, d))]
    # Loop through the label directories and collect the data in
    # two lists, labels and images.
    labels = []
    images = []
    for d in directories:
        label_dir = os.path.join(data_dir, d)
	
        file_names = [os.path.join(label_dir, f) 
                      for f in os.listdir(label_dir) if f.endswith(".ppm")]
        # For each label, load it's images and add them to the images list.
        # And add the label number (i.e. directory name) to the labels list.
        for f in file_names:
            images.append(skimage.data.imread(f))
            labels.append(int(d))
    return images, labels

#todo: add an evaluation set 
def prepare_data(images, labels):
    c_images = [skimage.transform.resize(image, (28, 28))
                    for image in images]
    c_images = np.array(c_images)
    #todo: can we do this in color?	
    c_images = rgb2gray(c_images)
    c_images = [image.flatten()
                    for image in c_images]
    c_labels = np.array(labels)
    c_labels = convertToOneHot(c_labels, 62)
    #perhaps tie the two arrays together in a batchable array
    return np.array(c_images), np.array(c_labels)

def convertToOneHot(vector, num_classes=None):
    """
    Converts an input 1-D vector of integers into an output
    2-D array of one-hot vectors, where an i'th input value
    of j will set a '1' in the i'th row, j'th column of the
    output array.

    Example:
        v = np.array((1, 0, 4))
        one_hot_v = convertToOneHot(v)
        print one_hot_v

        [[0 1 0 0 0]
         [1 0 0 0 0]
         [0 0 0 0 1]]
    """

    assert isinstance(vector, np.ndarray)
    assert len(vector) > 0

    if num_classes is None:
        num_classes = np.max(vector)+1
    else:
        assert num_classes > 0
        assert num_classes >= np.max(vector)

    result = np.zeros(shape=(len(vector), num_classes))
    result[np.arange(len(vector)), vector] = 1
    return result.astype(int)

def display_images_and_labels(images, labels):
    """Display the first image of each label."""
    unique_labels = set(labels)
    plt.figure(figsize=(15, 15))
    i = 1
    for label in unique_labels:
        # Pick the first image for each label.
        image = images[labels.index(label)]
        plt.subplot(8, 8, i)  # A grid of 8 rows x 8 columns
        plt.axis('off')
        plt.title("Label {0} ({1})".format(label, labels.count(label)))
        i += 1
        _ = plt.imshow(image)
    plt.show()

def display_label_images(images, label):
    """Display images of a specific label."""
    limit = 24  # show a max of 24 images
    plt.figure(figsize=(15, 5))
    i = 1

    start = labels.index(label)
    end = start + labels.count(label)
    for image in images[start:end][:limit]:
        plt.subplot(3, 8, i)  # 3 rows, 8 per row
        plt.axis('off')
        i += 1
        plt.imshow(image)
    plt.show()

def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)

def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


# Load training and testing datasets.
ROOT_PATH = "/traffic"
train_data_dir = "datasets/Training"#os.path.join(ROOT_PATH, "datasets/Training")
test_data_dir = "datasets/Testing" #os.path.join(ROOT_PATH, "datasets/Testing")

print("Preparing test and train data")
start_time = time.time()
train_images, train_labels = load_data(train_data_dir)
LABEL_SIZE = len(set(train_labels))
train_images, train_labels = prepare_data(train_images, train_labels)
train_data = DataSet(train_images, train_labels, reshape=False)

test_images, test_labels = load_data(test_data_dir)
test_images, test_labels = prepare_data(test_images, test_labels)
test_data = DataSet(test_images, test_labels, reshape=False)

print('Time taken ', time.time() - start_time)

#print("Unique Labels: {0}\nTotal Images: {1}".format(len(set(train_labels)), len(train_images)))

#display_images_and_labels(images32, labels)
#todo: move the construction to a function, add tensorboard logging
print("Start constructing the network")
#print("Constructing the network: Inputs")
start_time = time.time()
x = tf.placeholder(tf.float32, [None, 784])
x_image = tf.reshape(x, [-1,28,28,1]) #with color this was x, y, z, 3
y_ = tf.placeholder(tf.float32, shape=[None, LABEL_SIZE])

x_image = tf.cast(x_image, tf.float32);

#print("Constructing the network: Conv 1")
#convolution 1
W_conv1 = weight_variable([5, 5, 1, 32])
b_conv1 = bias_variable([32])

h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

#print("Constructing the network: Conv 2")
#convolution 2
W_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])

h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

#print("Constructing the network : Dense Layer")
#densely connected layer
W_fc1 = weight_variable([7 * 7 * 64, 1024])
b_fc1 = bias_variable([1024])

h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

#print("Constructing the network : Dropout")
#dropout to reduce overfitting
keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

#print("Constructing the network : Read Out ")
W_fc2 = weight_variable([1024, LABEL_SIZE])
b_fc2 = bias_variable([LABEL_SIZE])

y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

#define the training
#print("Constructing the network : Evaluation")
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y_conv, y_))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

print('Time taken ', time.time() - start_time)

print("Start the training")
start_time = time.time()
sess.run(tf.global_variables_initializer())
for i in range(20000): 
  batch = train_data.next_batch(50) 
  if i%100 == 0:
    train_accuracy = accuracy.eval(feed_dict={
        x:batch[0], y_: batch[1], keep_prob: 1.0})
    print("step %d, training accuracy %g"%(i, train_accuracy))
  train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})
print('Time taken ', time.time() - start_time)

print("Start evaluation")
start_time = time.time()
print("test accuracy %g"%accuracy.eval(feed_dict={
    x: test_data.images, y_: test_data.labels, keep_prob: 1.0}))
print('Time taken ', time.time() - start_time)



