import numpy as np
import scipy.misc as misc
import tensorflow as tf

a1 = np.array([1,2,3,4,5])
a2 = a1.tobytes()
a3 = np.fromstring(a2, dtype = 'int64')
a4 = tf.decode_raw(a2,out_type = 'int64')

img0 = misc.imread('1_00001.jpg')
img1 = tf.gfile.FastGFile('1_00001.jpg','rb').read()
img2 = tf.image.decode_jpeg(img1, channels=3)

with tf.Session() as sess:
  img3 = img2.eval()
  a5 = a4.eval()
