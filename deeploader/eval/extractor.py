# -*- coding:utf-8 -*-
from __future__ import print_function
import math
import numpy as np
import logging
import time
import os
import argparse
import cv2
import sys

from deeploader.eval.run_verify import *
        
        
def _extract_feature_each(extractor, img_list):
    feat_list = []
    n = len(img_list)
    idx = 1
    for img in img_list:
        feat = extractor.extract(img)
        feat_list.append(feat)
        if idx > 1:
            print('{}'.format('\b'*10))
        print('{}/{}'.format(idx, n), end='')
        idx += 1
    return feat_list


def _extract_feature_batch(extractor, pair_list, size = 0):
    batch_size = extractor.batch_size
    feat_list = []
    npairs = len(pair_list)
    if size == 0:
        size = npairs
    size = min(size, npairs)
    nbatch = (size + batch_size - 1) // batch_size

    for batch in range(nbatch):
        # make a batch
        x_list = []
        for i in range(batch_size):
            pairid = (batch * batch_size + i)
            if pairid >= npairs:
                pairid = npairs - 1
            x_list.append(pair_list[pairid])
        #
        x_batch = np.stack(x_list, axis=0)
        feat = extractor.extract(x_batch)
        
        for i in range(batch_size):
            a = feat[i,:]
            if len(feat_list) < size:
                feat_list.append(a)
    
    return feat_list 

    
def extract_list(extractor, img_list, size = 0):
    batch_size = extractor.batch_size
    if batch_size > 1:
        return _extract_feature_batch(extractor, img_list, size)
    return _extract_feature_each(extractor, img_list)
    
    
def crop_image_list(img_list, imsize):
    """
    crop images
    """
    out_list = []
    h, w, c = img_list[0].shape
    x1 = (w - imsize[0])/2
    y1 = (h - imsize[1])/2
    for pair in img_list:
        img1 = pair
        img1 = img1[y1:(y1+imsize[1]),x1:(x1+imsize[0]),:]
        out_list.append(img1)
    #print(img1.shape)
    return out_list
    

def norm_image_list(img_list):
    """
    norm images
    """
    out_list = []
    for pair in img_list:
        img1 = pair
        img1 = ( np.float32(img1) - 127.5 ) / 128
        out_list.append(img1)
    return out_list

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--type", help="caffe | tensorflow | mxnet")
    parser.add_argument("--test_set", help="lfw | ytf")
    parser.add_argument("--data",   help="lfw.np or pair.txt")
    parser.add_argument("--prefix", help="data prefix")
    parser.add_argument("--model_path", help= 'specify which model to test ')
    parser.add_argument('--image_size', default="112, 96", help='image size height, width')
    parser.add_argument('--net_depth', default=50, help='resnet depth, default is 50')
    parser.add_argument('--batch_size', default=64, help='batch size to train network')
    parser.add_argument("--model_name", help= 'specify which model to test \n'
                                              ' centerface\n'
                                              ' sphereface\n'
                                              ' AMSoftmax\n'
                                              ' arcface\n'
                                              ' yours \n')
    parser.add_argument("--dist_type", default='cosine', help="distance measure ['cosine', 'L2', 'SSD']")
    parser.add_argument("--do_mirror", default=False, help="mirror image and concatinate features")
    parser.add_argument("--do_norm", default=True, help="norm image before feed to nets")
    parser.add_argument("--embed_name", help= 'specify output blob name')
    args = parser.parse_args()
    return args

def build_extractor(args):
    image_size = args.image_size.split(',')
    image_size = (int(image_size[1]), int(image_size[0])) 
    model_name = args.model_name
    do_mirror = args.do_mirror
    if args.type == 'caffe':
        from plats.caffe.caffe_model_factory import model_factory
        extractor, image_size = model_factory(model_name, do_mirror)
        print('Testing model\t: %s' % (extractor.weight))
        # do norm
        args.do_norm = True
    elif args.type == 'tensorflow':
        from plats.tensorflow.resnet50_extractor import get_extractor
        image_size = [image_size[1], image_size[0]]
        extractor = get_extractor(args)
        # do norm
        args.do_norm = True
    elif args.type == 'mxnet':
        from plats.mxnet.mxnet_extractor import MxnetExtractor
        extractor = MxnetExtractor(args.model_path, args.batch_size, image_size, args.embed_name)
        
    args.image_size = image_size
    return extractor
    
if __name__ == '__main__':
    args = parse_args()
    output_dir = '.'
    # parse args   
    image_size = args.image_size.split(',')
    image_size = (int(image_size[1]), int(image_size[0])) 
    model_name = args.model_name
    test_set = args.test_set
    dist_type = args.dist_type
    do_mirror = args.do_mirror
    print('Dataset  \t: %s (%s,%s)' % (args.test_set, args.data, args.prefix))
    print('Testing  \t: %s' % model_name)
    print('Distance \t: %s' % dist_type)
    print('Do mirror\t: {}'.format(do_mirror))
    
    # load images
    if args.data.find('.np') > 0:
        pos_img, neg_img = pickle.load(open(args.data, 'rb'))
        #pos_img, neg_img = pickle.load(open(lfw_data, 'rb'), encoding='iso-8859-1')
    else:
        if args.test_set == 'lfw':
            pos_img, neg_img = load_image_paris(args.data, args.prefix)
        else:
            pos_img, neg_img = load_ytf_pairs(args.data, args.prefix)
        
    # crop image
    pos_img = crop_image_list(pos_img, image_size)
    neg_img = crop_image_list(neg_img, image_size)
    # model
    if args.type == 'caffe':
        from plats.caffe.caffe_model_factory import model_factory
        extractor, image_size = model_factory(model_name, do_mirror)
        print('Testing model\t: %s' % (extractor.weight))
        # do norm
        args.do_norm = True
    elif args.type == 'tensorflow':
        from plats.tensorflow.resnet50_extractor import get_extractor
        args.image_size = [image_size[1], image_size[0]]
        extractor = get_extractor(args)
        # do norm
        args.do_norm = True
    elif args.type == 'mxnet':
        from plats.mxnet.mxnet_extractor import MxnetExtractor
        extractor = MxnetExtractor(args.model_path, args.batch_size, image_size, args.embed_name)
    print('Image size\t: {}'.format(image_size))
    print('Do norm  \t: {}'.format(args.do_norm))
    if args.do_norm == True:
        print('Norm images')
        pos_img = norm_image_list(pos_img)
        neg_img = norm_image_list(neg_img)
    # compute feature
    print('Extracting features ...')
    pos_list = extract_feature(extractor, pos_img)
    print('  Done positive pairs')
    neg_list = extract_feature(extractor, neg_img)
    print('  Done negative pairs')

    # evaluate
    print('Evaluating ...')
    precision, std, threshold, pos, neg, _ = verification(pos_list, neg_list, dist_type = dist_type)    
    # _, title = os.path.split(extractor.weight)
    #draw_chart(title, output_dir, {'pos': pos, 'neg': neg}, precision, threshold)
    print('------------------------------------------------------------')
    print('Precision on %s : %1.5f+-%1.5f \nBest threshold   : %f' % (args.test_set, precision, std, threshold))
   
   

