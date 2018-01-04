import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import matplotlib.pyplot as plt
import cv2
import numpy as np
#img=cv2.imread('bfritzv1.pgm',0)
#cv2.imshow('h',img)
#cv2.waitKey(0)
#cv2.destroyAllWindows()
X = np.random.random((2,784))
Y=np.array([[1,0,0,0,0,0,0,0,0,0],[0,1,0,0,0,0,0,0,0,0]])
def init_weights(shape):
    init_random_dist = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(init_random_dist)
def init_bias(shape):
    init_bias_vals = tf.constant(0.1, shape=shape)
    return tf.Variable(init_bias_vals)
def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')
def max_pool_2by2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                          strides=[1, 2, 2, 1], padding='SAME')
def convolutional_layer(input_x, shape):
    W = init_weights(shape)
    b = init_bias([shape[3]])
    return tf.nn.relu(conv2d(input_x, W) + b)
def normal_full_layer(input_layer, size):
    input_size = int(input_layer.get_shape()[1])
    W = init_weights([input_size, size])
    b = init_bias([size])
    return tf.matmul(input_layer, W) + b
x = tf.placeholder(tf.float32,shape=[None,784])
y_true = tf.placeholder(tf.float32,shape=[None,10])
x_image = tf.reshape(x,[-1,28,28,1])
convo_1 = convolutional_layer(x_image,shape=[11,11,1,5])
convo_1_pooling = max_pool_2by2(convo_1)
convo_2 = convolutional_layer(convo_1_pooling,shape=[6,6,5,10])
convo_2_pooling = max_pool_2by2(convo_2)
convo_2_flat = tf.reshape(convo_2_pooling,[-1,7*7*10])
full_layer_one = tf.nn.relu(normal_full_layer(convo_2_flat,500))
hold_prob = tf.placeholder(tf.float32)
full_one_dropout = tf.nn.dropout(full_layer_one,keep_prob=hold_prob)
y_pred = normal_full_layer(full_one_dropout,10)
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_true,logits=y_pred))
optimizer = tf.train.AdamOptimizer(learning_rate=0.0001)
train = optimizer.minimize(cross_entropy)

init = tf.global_variables_initializer()

steps = 1

with tf.Session() as sess:
    sess.run(init)

    for i in range(steps):

        batch_x=X
        batch_y =Y
        sess.run(train,feed_dict={x:batch_x,y_true:batch_y,hold_prob:0.5})

        matches = tf.equal(tf.argmax(y_pred, 1), tf.argmax(y_true, 1))

        acc = tf.reduce_mean(tf.cast(matches, tf.float32))
        IN,OUT=sess.run([acc,full_one_dropout],feed_dict={x:batch_x,y_true:batch_y,hold_prob:0.5})



        # PRINT OUT A MESSAGE EVERY 100 STEPS

        print(IN)
        print(OUT)


