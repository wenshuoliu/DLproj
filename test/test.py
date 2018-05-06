import tensorflow as tf

a = tf.constant(5)
b = tf.constant(6)
sess = tf.Session()
sess.run(a+b)
sess.close()