import tensorflow as tf
import cv2
import numpy as np
from rnn import *
class model3:
    def __init__(self,s1,s2,u):
        self.num_inputs = 500
        self.outputs=0
        self.states=0
        self.s1=s1
        self.s2=s2
        self.u=u
        self.num_neurons = 500

        self.num_outputs = 1
        self.x = tf.placeholder(tf.float32, shape=[None, 784])
        self.y_true = tf.placeholder(tf.float32, shape=[None, 28])
        self.x_image = tf.reshape(self.x, [-1, 28, 28, 1])
        self.xx = tf.placeholder(tf.float32, [None, 28, self.num_inputs])

        self.yy = tf.placeholder(tf.float32, [None, 28,self.num_outputs])

        self.cell = tf.contrib.rnn.OutputProjectionWrapper(tf.contrib.rnn.BasicRNNCell(num_units=self.num_neurons, activation=tf.nn.relu,reuse=tf.get_variable_scope().reuse),
        output_size=self.num_outputs)


    def init_weights(self,shape):
        init_random_dist = tf.truncated_normal(shape, stddev=0.1)
        return tf.Variable(init_random_dist)
    def init_bias(self,shape):
        init_bias_vals = tf.constant(0.1, shape=shape)
        return tf.Variable(init_bias_vals)
    def conv2d(self,x, W):
        return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')
    def max_pool_2by2(self,x):
        return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                          strides=[1, 2, 2, 1], padding='SAME')
    def convolutional_layer(self,input_x, shape):
        W = self.init_weights(shape)
        b = self.init_bias([shape[3]])
        return tf.nn.relu(self.conv2d(input_x, W) + b)
    def normal_full_layer(self,input_layer, size):
        input_size = int(input_layer.get_shape()[1])
        W = self.init_weights([input_size, size])
        b = self.init_bias([size])
        return tf.matmul(input_layer, W) + b
    def run(self):


        convo_1 = self.convolutional_layer(self.x_image,shape=[11,11,1,5])
        convo_1_pooling = self.max_pool_2by2(convo_1)
        convo_2 = self.convolutional_layer(convo_1_pooling,shape=[6,6,5,10])
        convo_2_pooling = self.max_pool_2by2(convo_2)
        convo_2_flat = tf.reshape(convo_2_pooling,[-1,7*7*10])
        full_layer_one = tf.nn.relu(self.normal_full_layer(convo_2_flat,500))
        hold_prob = tf.placeholder(tf.float32)
        full_one_dropout = tf.nn.dropout(full_layer_one,keep_prob=hold_prob)

        init = tf.global_variables_initializer()
        y_pred = self.normal_full_layer(full_one_dropout,28)
        cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.y_true,logits=y_pred))
        optimizer = tf.train.AdamOptimizer(learning_rate=0.0001)
        train = optimizer.minimize(cross_entropy)

        steps = 1
        ii=np.zeros((1,28,500))
        Y=np.array([[1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0], [0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0],[1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1]])




        learning_rate = 0.0001

        num_train_iterations = 1

        batch_size = 1




        self.outputs, self.states = tf.nn.dynamic_rnn(self.cell, self.xx, dtype=tf.float32)

        Loss = tf.reduce_mean(tf.square(self.outputs - self.yy))

        Optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
        Train = Optimizer.minimize(Loss)

        init = tf.global_variables_initializer()
        saver = tf.train.Saver()


        l=np.zeros((1,28,500))



        X=[]
        for i in range(0,28):

            c=cv2.imread('1/New folder/3.'+str(i)+'.jpg',0)
            c=cv2.resize(c,(28,28))
            c=c/256
            c=c.flatten()

            X.append(c)

        X=np.array(X)



        with tf.Session() as sess:
            sess.run(init)
            saver.restore(sess, "./cnn3")

            for i in range(steps):

                batch_x, batch_y = X,Y
        #sess.run(train,feed_dict={x:batch_x,y_true:batch_y,hold_prob:0.5})
                matches = tf.equal(tf.argmax(y_pred, 1), tf.argmax(self.y_true, 1))

                acc = tf.reduce_mean(tf.cast(matches, tf.float32))



                s,IN = sess.run([acc,full_one_dropout], feed_dict={self.x: batch_x, self.y_true: batch_y, hold_prob: 0.5})
                loss = sess.run(cross_entropy, feed_dict={self.x: batch_x, self.y_true: batch_y, hold_prob: 0.5})

                print(s)
                print(loss)


                saver.save(sess, self.s1)
                if(i==0):
                    for j in range(28):
                        l[0][j]=IN[j]



                    with tf.Session() as s:
                        s.run(init)
                        saver.restore(sess,self.s2)

                        for iteration in range(num_train_iterations):

                            X_batch, y_batch = l,self.u
                    #sess.run(Train, feed_dict={XX: X_batch, yy: y_batch})


                            mse = Loss.eval(feed_dict={self.xx: X_batch, self.yy: y_batch})
                            matches = tf.equal(tf.argmax(self.outputs, 1), tf.argmax(self.yy, 1))

                            acc = tf.reduce_mean(tf.cast(matches, tf.float32))

                            v = sess.run(acc, feed_dict={self.xx: X_batch, self.yy: y_batch, hold_prob: 0.5})
                            V = sess.run(Loss, feed_dict={self.xx: X_batch, self.yy: y_batch, hold_prob: 0.5})
                            u=sess.run(self.outputs, feed_dict={self.xx: X_batch, self.yy: y_batch, hold_prob: 0.5})
                            j=tf.nn.softmax_cross_entropy_with_logits(labels=self.yy, logits=self.outputs)

                            print(v)
                            print(V)
                            print(sess.run(j, feed_dict={self.xx: X_batch, self.yy: y_batch}))
                            print(u)

                            saver.save(s, "./rnn3")
                    s.close()
        sess.close()

        return V

u=np.ones((1,28,1))*-1
m1=model3("./cnn3","./rnn3",u)
u = np.ones((1, 28, 1))

e=m1.run()
m2=model3("./cnn","./rnn_time_series_model",u)
e2=m2.run()
