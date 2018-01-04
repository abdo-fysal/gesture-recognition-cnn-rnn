import tensorflow as tf
import cv2
import numpy as np

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
y_true = tf.placeholder(tf.float32,shape=[None,28])
x_image = tf.reshape(x,[-1,28,28,1])
convo_1 = convolutional_layer(x_image,shape=[11,11,1,5])
convo_1_pooling = max_pool_2by2(convo_1)
convo_2 = convolutional_layer(convo_1_pooling,shape=[6,6,5,10])
convo_2_pooling = max_pool_2by2(convo_2)
convo_2_flat = tf.reshape(convo_2_pooling,[-1,7*7*10])
full_layer_one = tf.nn.relu(normal_full_layer(convo_2_flat,500))
hold_prob = tf.placeholder(tf.float32)
full_one_dropout = tf.nn.dropout(full_layer_one,keep_prob=hold_prob)

init = tf.global_variables_initializer()
y_pred = normal_full_layer(full_one_dropout,28)
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_true,logits=y_pred))
optimizer = tf.train.AdamOptimizer(learning_rate=0.0001)
train = optimizer.minimize(cross_entropy)
tf.add_to_collection('out', full_one_dropout)

steps = 1
ii=np.zeros((1,28,500))
Y=np.array([[1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0], [0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0],[1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1]])


num_inputs = 500

num_neurons = 500

num_outputs = 1

learning_rate = 0.0001

num_train_iterations = 1

batch_size = 1

XX = tf.placeholder(tf.float32, [None, 28, num_inputs])

yy = tf.placeholder(tf.float32, [None, 28, num_outputs])

cell = tf.contrib.rnn.OutputProjectionWrapper(
    tf.contrib.rnn.BasicRNNCell(num_units=num_neurons, activation=tf.nn.relu),
    output_size=num_outputs)

outputs, states = tf.nn.dynamic_rnn(cell, XX, dtype=tf.float32)

Loss = tf.reduce_mean(tf.square(outputs - yy))

Optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
Train = Optimizer.minimize(Loss)

init = tf.global_variables_initializer()
saver = tf.train.Saver()

u=np.ones((1,28,1))*2
U=np.ones((1,28,1))*0.3

l=np.zeros((1,28,500))


vidcap = cv2.VideoCapture(0)
success,c = vidcap.read()
count = 0
success = True
X=[]
ct1=np.zeros(c.shape)
ct2=np.zeros(c.shape)
ct3=np.zeros(c.shape)

delta=np.zeros(c.shape)
b=0
while count <28:
  success,c = vidcap.read()
  cv2.imshow('a',c)
  print('Read a new frame: ', success)
  count += 1
  c=cv2.cvtColor(c, cv2.COLOR_BGR2GRAY)
  c = cv2.resize(c, (28, 28))
  c = c / 256
  c = c.flatten()
  if(b==0):
      ct1=c
  elif(b==1):
      ct2=c
  elif(b==2):
      ct3=c
  else:
      delta=(ct2-ct1) and (ct3-ct2)
      ct1=ct2
      ct2=ct3
      ct3=c
  b=b+1


  X.append(delta)
cv2.destroyAllWindows()

X=np.array(X)

with tf.Session() as sess:
    saver.restore(sess,"./cnn_1")
    for i in range(steps):

        batch_x, batch_y = X,Y
        #sess.run(train,feed_dict={x:batch_x,y_true:batch_y,hold_prob:0.5})
        matches = tf.equal(tf.argmax(y_pred, 1), tf.argmax(y_true, 1))

        acc = tf.reduce_mean(tf.cast(matches, tf.float32))



        s,IN = sess.run([acc,full_one_dropout], feed_dict={x: batch_x, y_true: batch_y, hold_prob: 0.5})
        loss = sess.run(cross_entropy, feed_dict={x: batch_x, y_true: batch_y, hold_prob: 0.5})

        print(s)
        print(loss)


        if(i==0):
            for j in range(28):
                l[0][j]=IN[j]
                saver.save(sess, "./cnn_1")

            with tf.Session() as s:
                saver.restore(s,"./rnn_1")

                for iteration in range(num_train_iterations):

                    X_batch, y_batch = l,u
                    #sess.run(Train, feed_dict={XX: X_batch, yy: y_batch})


                    mse = Loss.eval(feed_dict={XX: X_batch, yy: y_batch})
                    matches = tf.equal(tf.argmax(outputs, 1), tf.argmax(yy, 1))

                    acc = tf.reduce_mean(tf.cast(matches, tf.float32))

                    v = sess.run(acc, feed_dict={XX: X_batch, yy: y_batch, hold_prob: 0.5})
                    V1 = sess.run(Loss, feed_dict={XX: X_batch, yy: y_batch, hold_prob: 0.5})
                    u=sess.run(outputs, feed_dict={XX: X_batch, yy: y_batch, hold_prob: 0.5})
                    j=tf.nn.softmax_cross_entropy_with_logits(labels=yy, logits=outputs)

                    print(v)
                    print(V1)
                    print(sess.run(j, feed_dict={XX: X_batch, yy: y_batch}))
                    print(u)
                    saver.save(sess, "./rnn_1")

with tf.Session() as sess:
    sess.run(init)

    for i in range(steps):

        batch_x, batch_y = X,Y

        #sess.run(train,feed_dict={x:batch_x,y_true:batch_y,hold_prob:0.5})
        matches = tf.equal(tf.argmax(y_pred, 1), tf.argmax(y_true, 1))

        acc = tf.reduce_mean(tf.cast(matches, tf.float32))



        s,IN = sess.run([acc,full_one_dropout], feed_dict={x: batch_x, y_true: batch_y, hold_prob: 0.5})
        loss = sess.run(cross_entropy, feed_dict={x: batch_x, y_true: batch_y, hold_prob: 0.5})

        print(s)
        print(loss)


        if(i==0):
            for j in range(28):
                l[0][j]=IN[j]
                saver.save(sess, "./cnn_0")

            with tf.Session() as s:
                s.run(init)

                for iteration in range(num_train_iterations):

                    X_batch, y_batch = l,U

                    #sess.run(Train, feed_dict={XX: X_batch, yy: y_batch})


                    mse = Loss.eval(feed_dict={XX: X_batch, yy: y_batch})
                    matches = tf.equal(tf.argmax(outputs, 1), tf.argmax(yy, 1))

                    acc = tf.reduce_mean(tf.cast(matches, tf.float32))

                    v = sess.run(acc, feed_dict={XX: X_batch, yy: y_batch, hold_prob: 0.5})
                    V2 = sess.run(Loss, feed_dict={XX: X_batch, yy: y_batch, hold_prob: 0.5})
                    u=sess.run(outputs, feed_dict={XX: X_batch, yy: y_batch, hold_prob: 0.5})
                    j=tf.nn.softmax_cross_entropy_with_logits(labels=yy, logits=outputs)

                    print(v)
                    print(V2)
                    print(sess.run(j, feed_dict={XX: X_batch, yy: y_batch}))
                    print(u)
                    saver.save(sess, "./rnn_0")



if(V1<V2):
    im=cv2.imread('21.6.jpg')
    cv2.imshow('class1',im)
    cv2.waitKey(0)

else:
    im = cv2.imread('2.6.jpg')
    cv2.imshow('class0', im)
    cv2.waitKey(0)
