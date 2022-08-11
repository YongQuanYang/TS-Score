from __future__ import print_function
import os
import sys

import tensorflow as tf
import tensorflow.contrib.slim as slim

import cv2
import imageio
import numpy as np
import copy

from builders import model_builder

os.environ['CUDA_VISIBLE_DEVICES']='0'


session_config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
session_config.gpu_options.allow_growth = True
sess = tf.Session(config=session_config)
net_input = tf.placeholder(tf.float32, shape=[None, None, None, 3])
network, init_fn = model_builder.build_model(model_name='TinyFCN', net_input=net_input, num_classes=2, crop_width=256, crop_height=256, is_training=False)
network = tf.nn.softmax(network)

restore_saver=tf.train.Saver()
sess.run(tf.global_variables_initializer())
restore_saver.restore(sess, './ckpt/model.ckpt')

imgp = './testing_data/img'
stroma_path = './pred_4_pCR'

if not os.path.exists(stroma_path):
    os.makedirs(stroma_path)

imgs = os.listdir(imgp)
for img in imgs:
    image_path = os.path.join(imgp, img)
    image = cv2.cvtColor(cv2.imread(image_path,-1), cv2.COLOR_BGR2RGB)
    image = image / 255.0
    image = image[np.newaxis, :, :, :]
    output_image = sess.run(network, feed_dict={net_input: image})
    output_image = np.array(output_image[0, :, :, :])
    
    image = imageio.imread(image_path)
    label = np.argmax(output_image, -1)
    label_img = copy.copy(label)*0
    label_img[:, :][label == 1] = 255
    label_img[:, :][label == 0] = 0
    label_img = label_img.astype('uint8')
    k = np.ones((5,5), np.uint8)
    label_img = cv2.morphologyEx(label_img, cv2.MORPH_OPEN, k)
    label_img[label_img>0] = 2
    label_img[label_img==0] = 1
    label_img[label_img==2] = 0
    binary_label = label_img
    image[:,:,0][binary_label==0] = 0
    image[:,:,1][binary_label==0] = 0
    image[:,:,2][binary_label==0] = 0
	
    imageio.imwrite(os.path.join(stroma_path, img.split('.')[0] + '.png'), image)




