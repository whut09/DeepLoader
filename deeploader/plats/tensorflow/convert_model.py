# -*- coding:utf-8 -*-
from __future__ import print_function
import math
import numpy as np
import logging
import time
import sys
import os
import argparse
import cv2
import tensorflow as tf
import tensorlayer as tl
import tensorflow.contrib.slim as slim
from tensorflow.core.protobuf import config_pb2

from nets.L_Resnet_E_IR_fix_issue9 import get_resnet


def makedirs(path):
    dir,fname = os.path.split(path)
    if not os.path.exists(dir):
        try:
            os.makedirs(dir)
        except:
            pass
            
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", help= 'specify which model to convert ')
    parser.add_argument("--output", default='./deploy', help= 'specify which model to convert ')
    parser.add_argument('--image_size', default=[112, 96], help='image size height, width')
    parser.add_argument('--net_depth', default=50, help='resnet depth, default is 50')

    args = parser.parse_args()
    images = tf.placeholder(name='img_inputs', shape=[None, args.image_size[0], args.image_size[1], 3], dtype=tf.float32)
    dropout_rate = tf.placeholder(name='dropout_rate', dtype=tf.float32)

    print('Buiding net structure')
    w_init_method = tf.contrib.layers.xavier_initializer(uniform=False)
    #net = get_resnet(images, args.net_depth, type='ir', w_init=w_init_method, trainable=True, keep_rate=dropout_rate)
    
    # test net  because of batch normal layer
    tl.layers.set_name_reuse(True)
    test_net = get_resnet(images, args.net_depth, type='ir', w_init=w_init_method, trainable=False, reuse=tf.AUTO_REUSE, keep_rate=dropout_rate)
    # 3.10 define sess
    #sess = tf.Session()
    gpu_config = tf.ConfigProto(allow_soft_placement=True )
    gpu_config.gpu_options.allow_growth = True

    sess = tf.Session(config=gpu_config)
    # 3.13 init all variables
    sess.run(tf.global_variables_initializer())
    # restore weights
    saver = tf.train.Saver()
    saver.restore(sess, args.model_path)
    _, fname = os.path.split(args.model_path)
    filename = os.path.join(args.output, fname)
    makedirs(filename)
    saver.save(sess, filename)

