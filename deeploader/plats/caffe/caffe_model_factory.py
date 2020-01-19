# -*- coding:utf-8 -*-
from __future__ import print_function
import math
import numpy as np
import os

import sys
sys.path.insert(0, '../facealign')
sys.path.insert(0, '../util') 
  
from caffe_extractor import CaffeExtractor

def model_centerface(do_mirror):
    model_dir = './models/centerface/'
    model_proto = model_dir + 'face_deploy.prototxt'
    model_path = model_dir + 'face_model.caffemodel'
    image_size = (96, 112)
    extractor = CaffeExtractor(model_proto, model_path, do_mirror = do_mirror, featLayer='fc5')
    return extractor, image_size
    
def model_sphereface(do_mirror):
    model_dir = './models/sphereface/'
    model_proto = model_dir + 'sphereface_deploy.prototxt'
    model_path = model_dir + 'sphereface_model.caffemodel'
    image_size = (96, 112)
    extractor = CaffeExtractor(model_proto, model_path, do_mirror = do_mirror, featLayer='fc5')
    return extractor, image_size
    
def model_AMSoftmax(do_mirror):
    model_dir = './models/AMSoftmax/'
    if do_mirror:
        model_proto = model_dir + 'face_deploy_mirror_normalize.prototxt'
    else:
        model_proto = model_dir + 'deploy.prototxt'
    model_path = model_dir + 'face_train_test_iter_30000.caffemodel'
    image_size = (96, 112)
    extractor = CaffeExtractor(model_proto, model_path, do_mirror = False, featLayer='fc5')
    return extractor, image_size
    
    
def model_arcface(do_mirror):
    model_dir = './models/arcface/'
    model_proto = model_dir + 'model.prototxt'
    model_path = model_dir + 'model-r50-am.caffemodel'
    image_size = (112, 112)
    extractor = CaffeExtractor(model_proto, model_path, do_mirror = do_mirror, featLayer='fc1')
    return extractor, image_size
    

def model_mobileface(do_mirror):
    model_dir = './models/mobilefacenet/'
    model_proto = model_dir + 'mobilefacenet-res2-6-10-2-dim128-opencv.prototxt'
    model_path = model_dir + 'mobilefacenet-res2-6-10-2-dim128.caffemodel'
    image_size = (112, 112)
    extractor = CaffeExtractor(model_proto, model_path, do_mirror = do_mirror, featLayer='fc1')
    return extractor, image_size

        
def model_mobileface2(do_mirror):
    model_dir = './models/mobilefacenet/'
    model_proto = model_dir + 'model.prototxt'
    model_path = model_dir + 'model.caffemodel'
    image_size = (112, 112)
    extractor = CaffeExtractor(model_proto, model_path, do_mirror = do_mirror, featLayer='fc1')
    return extractor, image_size

 
def model_factory(name, do_mirror):
    model_dict = {
        'centerface':model_centerface, 
        'sphereface':model_sphereface, 
        'AMSoftmax' :model_AMSoftmax, 
        'arcface'   :model_arcface,
        'mobileface':model_mobileface, 
        'mobileface2':model_mobileface2
    }
    model_func = model_dict[name]
    return model_func(do_mirror)

