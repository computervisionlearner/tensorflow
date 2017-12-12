import tensorflow as tf
import numpy as np


with tf.variable_scope('fc3') as scope:
  fc3w = tf.get_variable(initializer = tf.truncated_normal([3],dtype=tf.float32, stddev=1e-1),trainable=True, name='weights')
  fc3b = tf.get_variable(initializer = tf.constant(1.0, shape=[3], dtype=tf.float32),trainable=True, name='biases')
  op = tf.add(fc3w, fc3b)

a = np.array([1,2,3])

sess = tf.Session()
sess.run(tf.global_variables_initializer())
result1 = sess.run(op)

with tf.variable_scope('fc3' ,reuse = True):
  sess.run(tf.get_variable('weights').assign(a))
  sess.run(tf.get_variable('biases').assign(a))
  result2 = sess.run(op)
  variable_name = [variable.name for variable in tf.global_variables()]

sess.close()
