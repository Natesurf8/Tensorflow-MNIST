'''
Created on Nov 27, 2017

@author: Peters
'''
print("initializing . . .")
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data", one_hot=True)
import tensorflow as tf


def weight(shape):
	return tf.Variable(tf.truncated_normal(shape, stddev=0.1), name="weight")
def bias(init, shape):
	return tf.Variable(tf.constant(init, tf.float32, shape), name="bias")


def conv2d(x, W):
	return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME', name="convolve")
def max_pool_2x2(x):
	return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name="Max_Pool_2x2")


lr = 0.002
channels_conv1 = 16
channels_conv2 = 32
patch_conv1 = 5
patch_conv2 = 5
channels_fc1 = 128
keep_prob_fc1 = 0.5

def calc_learn_rate(batch):
	return lr/3 + lr/(0.5 + batch/200)

# Input and Output:
with tf.name_scope('Input_Processing'):
	x = tf.placeholder(tf.float32, [None, 784], name="In_Vectors")
	x_image = tf.reshape(x, [-1, 28, 28, 1], "In_Images")

# First Layer:
with tf.name_scope('Conv1'):
	W_conv1 = weight([patch_conv1, patch_conv1, 1, channels_conv1])
	b_conv1 = bias(0, [channels_conv1])
	h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
	h_pool1 = max_pool_2x2(h_conv1)

# Second Layer:
with tf.name_scope('Conv2'):
	W_conv2 = weight([patch_conv2, patch_conv2, channels_conv1, channels_conv2])
	b_conv2 = bias(0, [channels_conv2])
	h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2, "Conv2")
	h_pool2 = max_pool_2x2(h_conv2)
	h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * channels_conv2], "Flat_Conv2")

# Third Layer:
with tf.name_scope('FC1'):
	W_fc1 = weight([7 * 7 * channels_conv2, channels_fc1])
	b_fc1 = bias(0.1, [channels_fc1])
	h_fc1 = tf.nn.relu_layer(h_pool2_flat, W_fc1, b_fc1, "fc1")
	
	keep_prob = tf.placeholder(tf.float32, name="keep_prob")
	h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob, name="fc1_dropout")

# Fourth Layer:
with tf.name_scope('Readout'):
	y = h_fc1_drop @ weight([channels_fc1, 10]) + bias(0.1, [10])

# Predictions and Data Accuracy
with tf.name_scope("accuracy"):
	y_ = tf.placeholder(tf.float32, [None, 10], "Labels")
	correct_prediction = tf.equal(
		tf.argmax(y, 1, name="guess_index"),
		tf.argmax(y_, 1, name="label_index"),
		name="predictions_correct")
	accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name="percent_accuracy")

# Train functions
with tf.name_scope('Training'):
	error = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y), name="error")
	learn_rate = tf.placeholder(tf.float32, name="learning_rate")
	train_step = tf.train.AdamOptimizer(learn_rate).minimize(error, name="GD_Optimizer")

# merged summaries
error_sm = tf.summary.scalar("Error", error)
train_acc_sm = tf.summary.scalar("Train", accuracy)
learn_rate_sm = tf.summary.scalar("Learn Rate", learn_rate)

"""
with tf.name_scope("Wc1_Vis"):
	w_min = tf.reduce_min(W_conv1)
	w_max = tf.reduce_max(W_conv1)
	with tf.name_scope("F_Norm"):
		w_norm = (W_conv1 - w_min) / (w_max - w_min)
	
	w_transposed = tf.transpose(W_conv1, [3, 0, 1, 2])
	filters_conv1 = tf.summary.image("conv1/filters", w_transposed, max_outputs=16)
"""

merged = tf.summary.merge([error_sm, train_acc_sm, learn_rate_sm])


# non-merged summaries
test_accuracy = tf.placeholder(tf.float32, name="Test_Accuracy")
test_acc_sm = tf.summary.scalar("Test SM", test_accuracy)
# test_acc_sm needs its own placeholder because it is dependent on python function;
# results of function are inputted to the placeholder
def dataset_accuracy(dataset, num_images):
	n = 1000
	assert num_images % n == 0
	score = 0;
	for i in range(0, int(num_images / n)):
		score += accuracy.eval(feed_dict={
			x: dataset.images[n * i: n * (i+1)],
			y_: dataset.labels[n * i: n * (i+1)],
			keep_prob: 1.0
		})
	return score / (num_images / n)


with tf.Session() as sess:
	train_writer = tf.summary.FileWriter("Summaries/train", sess.graph)
	sess.run(tf.global_variables_initializer())
	
	for i in range(1, 2001):
		batch = mnist.train.next_batch(50)
		train_step.run(feed_dict={
			x: batch[0],
			y_: batch[1],
			keep_prob: keep_prob_fc1,
			learn_rate: calc_learn_rate(i)
		})
		
		""" """"""
		--output--
		"""""" """
		if i % 200 == 0:
			train_writer.add_summary(sess.run(test_acc_sm, feed_dict={
				test_accuracy: dataset_accuracy(mnist.test, 10000)
			}), i)
			
		if i % 50 == 0:
			train_writer.add_summary(merged.eval(feed_dict={
				x: batch[0],
				y_: batch[1],
				keep_prob: keep_prob_fc1,
				learn_rate: calc_learn_rate(i)
			}), i)
			
			train_writer.flush()
			print("Mini-Batch #: " + str(i))