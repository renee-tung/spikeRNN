'''
CONSTRUCT GRAPH
'''

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

# input & output nodes
x = tf.placeholder(tf.float32, shape=[4,2], name='input')
y = tf.placeholder(tf.float32, shape=[4,1], name='prediction')

# connectivity weight matrices
w_HI = tf.Variable(tf.random_normal((2,2)), name='input_to_hidden')
w_OH = tf.Variable(tf.random_normal((2,1)), name='hidden_to_output')

# initialize bias terms
b_12 = tf.Variable(tf.zeros((2, )), name='biases1_2')
b_3 = tf.Variable(tf.zeros((1, )), name='bias3')

# forward pass
o_ = tf.sigmoid(tf.matmul(x, w_HI) + b_12)
o  = tf.sigmoid(tf.matmul(o_, w_OH) + b_3)

# loss function
L = y*tf.log(o) + (1-y)*tf.log(1-o)
loss = tf.reduce_mean(-L)

# optimizer & train
lr = 0.01
optimizer = tf.train.GradientDescentOptimizer(learning_rate = lr)
training_op = optimizer.minimize(loss)

'''
TRAIN GRAPH
'''

import numpy as np

# four possible input cases
X = np.zeros((4,2))
X[1,1] = 1
X[2,0] = 1
X[3,:] = 1

# output labels
Y = np.zeros((4,1))
Y[1:3] = 1

# activate TensorFlow session
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

# train over multiple trials
trained = False
counter = 0
while trained == False:
    _, out, lo = sess.run([training_op, o, loss], feed_dict = {x: X, y: Y})
    if counter%5000 == 0:
        print(lo)
        if lo < 0.10:
            trained = True
            break
    counter += 1
print(out)
