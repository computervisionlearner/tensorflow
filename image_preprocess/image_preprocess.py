#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  5 20:30:21 2017
@author: no1
"""

import tensorflow as tf
import matplotlib.pyplot as plt
import random
import numpy as np
images = []

def distort_color(image):
  image = tf.image.random_brightness(image,max_delta=32./255)
  image = tf.image.random_saturation(image,lower=0.5,upper=1.5)
  image = tf.image.random_hue(image,max_delta=0.2)
  image = tf.image.random_contrast(image,lower=0.5,upper=1.5)
  return tf.clip_by_value(image,0,1)


def crop_or_pad(image):
  '''
  如果原始尺寸的大小大于目标图像，那么这个函数会自动截取图像居中部分
  否则，函数会自动向四周填充0
  tf.image.crop_to_bounding_box,tf.image.pad_to_bounding_box 可以调整剪切或pad位置
  
  '''
  resized = tf.image.resize_images(image, size = (300,300), method= 0)
  resized1 = tf.image.resize_image_with_crop_or_pad(resized,150,150)
  resized2 = tf.image.resize_image_with_crop_or_pad(resized,400,400)
  resized3 = tf.image.central_crop(image,0.5)
  return resized1, resized2, resized3

def flip(image):
  flipped1 = tf.image.flip_up_down(image)
  flipped2 = tf.image.flip_left_right(image)
  flipped3 = tf.image.transpose_image(image)
  return flipped1, flipped2, flipped3

def draw_box(image):
  batched = tf.expand_dims(image, 0)
  box1 = tf.reshape([0.05,0.5,0.9,0.9],shape = (1,1,4))
  box2 = tf.reshape([0.35,0.47,0.5,0.56],shape = (1,1,4))
  boxes = tf.concat([box1,box2],axis = 1)

  result = tf.image.draw_bounding_boxes(batched, boxes)
  
  return result
  
with tf.gfile.FastGFile('orange.jpg','rb') as f:
  encode_img = f.read()
  
decoded_img = tf.image.decode_jpeg(encode_img)

decoded_img = tf.image.convert_image_dtype(decoded_img, dtype = tf.float32)#将uint8转换成[0,1.]之间的浮点型
distorted_img = distort_color(decoded_img)
center_crop, center_pad, center_crop_rate = crop_or_pad(decoded_img) 
up_down, left_right, trans = flip(decoded_img)
boxes = draw_box(decoded_img)
images.extend([decoded_img,distorted_img,center_crop,center_pad,center_crop_rate,up_down,left_right,trans,boxes])


with tf.Session() as sess:

  results = sess.run(images)
  fig, axes = plt.subplots(figsize=(12,12), nrows=3, ncols=3)
  for ax, img in zip(axes.flatten(), results): 
    ax.imshow(np.squeeze(img))
plt.show()
