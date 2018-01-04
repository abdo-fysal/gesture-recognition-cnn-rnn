import tensorflow as tf

from tensorflow.examples.tutorials.mnist import input_data
import numpy as np

from matplotlib import pyplot as plt
num_inputs=2
num_neurons=3
x0=tf.placeholder(tf.float32,(None,num_inputs))
x1=tf.placeholder(tf.float32,(None,num_inputs))
wx=tf.Variable(tf.random_normal(shape=[num_inputs,num_neurons]))
wy=tf.Variable(tf.random_normal(shape=[num_neurons,num_neurons]))
b=tf.Variable(tf.zeros(shape=[1,num_neurons]))
y0=tf.tanh(tf.matmul(x0,wx)+b)
y1=tf.tanh(tf.matmul(y0,wy)+tf.matmul(x1,wx)+b)
init=tf.global_variables_initializer()
x0_batch=np.array([[0,1],[2,3],[4,5]])
x1_batch=np.array([[100,101],[102,103],[104,105]])
with tf.Session() as ss:
    ss.run(init)
    out1,out2=ss.run([y0,y1],feed_dict={x0:x0_batch,x1:x1_batch})
print(out1)
print(out2)


