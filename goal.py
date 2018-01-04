import tensorflow as tf

import cv2
import numpy as np
class ImportGraph_rnn():
    """  Importing and running isolated TF graph """

    def __init__(self, loc):
        # Create local graph and use it in the session
        self.graph = tf.Graph()
        self.sess = tf.Session(graph=self.graph)
        self.XX = tf.placeholder(tf.float32, [None, 28, 500])
        self.hold_prob = tf.placeholder(tf.float32)
        self.u=np.ones((1,28,1))


        self.yy = tf.placeholder(tf.float32, [None, 28, 1])
        with self.graph.as_default():
            # Import saved model from location 'loc' into local graph
            saver = tf.train.import_meta_graph(loc + '.meta',
                                               clear_devices=True)
            saver.restore(self.sess, loc)
            # Get activation function from saved collection
            # You may need to change this in case you name it differently
            self.loss = tf.get_collection('Loss')

    def run(self, data):
        """ Running the activation function previously imported """
        # The 'x' corresponds to name of input placeholder
        return self.sess.run(self.loss, feed_dict={self.XX: data, self.yy: self.u, self.hold_prob: 0.5})

class ImportGraph_cnn():
    """  Importing and running isolated TF graph """
    def init_weights(self, shape):
        init_random_dist = tf.truncated_normal(shape, stddev=0.1)
        return tf.Variable(init_random_dist)

    def init_bias(self, shape):
        init_bias_vals = tf.constant(0.1, shape=shape)
        return tf.Variable(init_bias_vals)

    def conv2d(self, x, W):
        return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

    def max_pool_2by2(self, x):
        return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                                  strides=[1, 2, 2, 1], padding='SAME')

    def convolutional_layer(self, input_x, shape):
        W = self.init_weights(shape)
        b = self.init_bias([shape[3]])
        return tf.nn.relu(self.conv2d(input_x, W) + b)

    def normal_full_layer(self, input_layer, size):
        input_size = int(input_layer.get_shape()[1])
        W = self.init_weights([input_size, size])
        b = self.init_bias([size])
        return tf.matmul(input_layer, W) + b


    def __init__(self, loc):
        # Create local graph and use it in the session
        self.graph = tf.Graph()
        self.sess = tf.Session(graph=self.graph)
        self.Y=np.array([[1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0], [0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0],[1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1]])




        with self.graph.as_default():
            self.x = tf.placeholder(tf.float32, shape=[None, 784], name='x')
            self.y_true = tf.placeholder(tf.float32, shape=[None, 28])
            self.hold_prob = tf.placeholder(tf.float32)

            # Import saved model from location 'loc' into local graph
            saver = tf.train.import_meta_graph(loc + '.meta',
                                               clear_devices=True)
            saver.restore(self.sess, loc)
            x_image = tf.reshape(self.x, [-1, 28, 28, 1])

            # Get activation function from saved collection
            # You may need to change this in case you name it differently
            convo_1 = self.convolutional_layer(x_image, shape=[11, 11, 1, 5])
            convo_1_pooling = self.max_pool_2by2(convo_1)
            convo_2 = self.convolutional_layer(convo_1_pooling, shape=[6, 6, 5, 10])
            convo_2_pooling = self.max_pool_2by2(convo_2)
            convo_2_flat = tf.reshape(convo_2_pooling, [-1, 7 * 7 * 10])
            full_layer_one = tf.nn.relu(self.normal_full_layer(convo_2_flat, 500))
            hold_prob = tf.placeholder(tf.float32)
            self.init = tf.global_variables_initializer()

            self.full_one_dropout = tf.nn.dropout(full_layer_one, keep_prob=hold_prob, name='n')


    def run(self, data):
        """ Running the activation function previously imported """
        # The 'x' corresponds to name of input placeholder
        self.sess.run(self.init)
        return self.sess.run(self.full_one_dropout, feed_dict={self.x: data,self.y_true: self.Y, self.hold_prob: 0.5})




vidcap = cv2.VideoCapture('1/M_29982.avi')
success, c = vidcap.read()
count = 0
success = True
X = []
while count < 28:
    success, c = vidcap.read()
    print('Read a new frame: ', success)

    count += 1
    c = cv2.cvtColor(c, cv2.COLOR_BGR2GRAY)
    c = cv2.resize(c, (28, 28))
    c = c / 256
    c = c.flatten()

    X.append(c)

X = np.array(X)
### Using the class ###
data = X  # random data
m1=ImportGraph_cnn('cnn3')
v=m1.run(X)
l = np.zeros((1, 28, 500))
print(v)
for j in range(28):
    l[0][j] = v[j]



model = ImportGraph_rnn('rnn3')
result = model.run(l)
print(result)