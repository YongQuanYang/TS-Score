from __future__ import print_function
import os
import sys

import tensorflow as tf
import tensorflow.contrib.slim as slim

import cv2
import imageio
import numpy as np

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
gtp = './testing_data/gt'
visual_path = './testing_data/pred_4_VisualCmp'

if not os.path.exists(visual_path):
    os.makedirs(visual_path)

imgs = os.listdir(imgp)
for img in imgs:
    image_path = os.path.join(imgp, img)
    image = cv2.cvtColor(cv2.imread(image_path,-1), cv2.COLOR_BGR2RGB)
    image = image / 255.0
    image = image[np.newaxis, :, :, :]
    output_image = sess.run(network, feed_dict={net_input: image})
    output_image = np.array(output_image[0, :, :, :])
    
    image = imageio.imread(image_path)
    probs = output_image[:,:,-1]
    pred = probs * 255
    pred = pred.astype(np.uint8)
    pred[pred<127]=0
    blank = np.zeros((256,256,3))
    blank[:,:,1]=pred
    blank = blank.astype(np.uint8)
    image1 = image.copy()
    image[:,:,0][pred>127]=0
    image[:,:,1][pred>127]=0
    image[:,:,2][pred>127]=0
    image1[:,:,0][pred<=127]=0
    image1[:,:,1][pred<=127]=0
    image1[:,:,2][pred<=127]=0
    target_mask = image + image1 * 0.7 + blank * 0.3
    target_mask = np.array(target_mask,np.uint8)
    
    targetp = os.path.join(gtp, img)	
    target = cv2.imread(targetp, -1)
    contours = cv2.findContours(target, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    contours = contours[0]
    cv2.drawContours(target_mask, contours, -1, (0, 255, 0), 4)
	
    imageio.imwrite(os.path.join(visual_path, img.split('.')[0] + '_pred-gt.png'), target_mask)




