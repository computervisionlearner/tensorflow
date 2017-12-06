import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
#注意：tf.image.sample_distorted_bounding_box每次产生的返回值都不一样，都是随机的
with tf.gfile.FastGFile('orange.jpg','rb') as f:
  encode_img = f.read()

results = []
decoded_img = tf.image.decode_jpeg(encode_img, channels=3)
'''
tf.image.per_image_standardization(image)，此函数的运算过程是将整幅图片标准化（不是归一化），
加速神经网络的训练。主要有如下操作，(x - mean) / adjusted_stddev，其中x为图片的RGB三通道像素值，
mean分别为三通道像素的均值，adjusted_stddev = max(stddev, 1.0/sqrt(image.NumElements()))。
stddev为三通道像素的标准差，image.NumElements()计算的是三通道各自的像素个数。 
img = tf.image.per_image_standardization(decoded_img)
'''
img = tf.image.convert_image_dtype(decoded_img, tf.float32)
croped_image = tf.random_crop(img, size = [300,300,3])
boxes = tf.constant([[[0.05,0.05,0.9,0.7],[0.35,0.47,0.5,0.56]]])

begin, size, bbox_for_draw = tf.image.sample_distorted_bounding_box(tf.shape(croped_image),
                                                                    bounding_boxes=boxes,min_object_covered=0.1)
batched = tf.expand_dims(croped_image, 0)
image_with_box = tf.image.draw_bounding_boxes(batched, bbox_for_draw)
distorted_img = tf.slice(croped_image, begin, size)
#distorted_img和distorted_img1是一样的。
distorted_img1 = tf.image.crop_to_bounding_box(img, begin[0], begin[1], size[0], size[1])
results.extend([decoded_img, croped_image, image_with_box, distorted_img])
with tf.Session() as sess:
  results_ = sess.run(results)
  b,s = sess.run([begin,size])
  fig, axes = plt.subplots(figsize=(12,12), nrows=2, ncols=2)
  for ax, img in zip(axes.flatten(), results_): 
    ax.imshow(np.squeeze(img))
  plt.show()
