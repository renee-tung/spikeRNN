# import tensorflow as tf
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()


# Construct a computational graph
input_1 = tf.constant(3, name='input_1')
input_2 = tf.constant(10, name='input_2')
input_3 = tf.constant(12, name='input_3')
# prod = tf.multiply(input_1, input_2, name='Mult_op')
# input_sum = input_1 + input_2 + input_3
input_sum = tf.math.add_n([input_1, input_2, input_3])

# Run the above graph in a TF session
with tf.Session() as sess:
    writer = tf.summary.FileWriter("tf_example", sess.graph)
    print(sess.run(input_sum))
    writer.close()
