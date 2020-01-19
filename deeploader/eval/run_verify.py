# -*- coding:utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import math
import numpy as np
import logging
import time
import os
import pickle
import argparse
import cv2
import sys
import re
from tqdm import tqdm
import sys

from deeploader.util.verification import verification, compute_distance
from deeploader.util.distance import get_distance 
from deeploader.util.opencv import cvCopy
from deeploader.util.fileutil import makedirs

def parse_line(line):
    line = line.strip()
    # [0|1] path1, path2
    exp = r'^(\d),([^,]*),([^,]*)'
    pattern = re.compile(exp, re.I)   # re.I Ignore case
    m = pattern.match(line)
    if m:
        issame = int(m.group(1))
        issame = True if issame == 1 else False
        rel_path1 = m.group(2).strip()
        rel_path2 = m.group(3).strip()
        return issame, rel_path1, rel_path2
    # lfw    
    splits = line.split()
    # skip line
    if len(splits) < 3:
        return None
    
    # name id1 id2
    if len(splits) == 3:
        pair = (True, splits[0], splits[1], splits[0], splits[2])
    # name1 id1 name2 id2
    else:
        pair = (False, splits[0], splits[1], splits[2], splits[3])

    rel_path1 = '%s/%s_%04d.jpg' % (pair[1], pair[1], int(pair[2]))
    rel_path2 = '%s/%s_%04d.jpg' % (pair[3], pair[3], int(pair[4]))
    return pair[0], rel_path1, rel_path2
    
    
def load_image_list(pair_list):    
    img_list = []
    for pair in pair_list:
        # skip invalid pairs
        if not os.path.exists(pair[0]) or not os.path.exists(pair[1]):
            continue
        img1 = cv2.imread(pair[0])
        #img1 = cv2.cvtColor(img1, cv2.COLOR_RGB2BGR)
        img2 = cv2.imread(pair[1])
        #img2 = cv2.cvtColor(img2, cv2.COLOR_RGB2BGR)
        #print(img1.shape)
        img_list.append([img1, img2, pair[0], pair[1]])
    return img_list
    
def load_ytf_pairs(path, prefix):
    pos_list_ = []
    neg_list_ = []
    with open(path, 'r') as f:
        for line in f.readlines():
            line = line.strip()
            flag, a, b = line.split(',')
            flag = int(flag)
            a = os.path.join(prefix, a)
            b = os.path.join(prefix, b)
            if flag == 1:
                pos_list_.append([a, b])
            else:
                neg_list_.append([a, b])
                
    pos_img = load_image_list(pos_list_)
    neg_img = load_image_list(neg_list_)
    return pos_img, neg_img
    
    
def load_image_paris(pair_path, prefix):
    pair_list = []
    # parse pairs
    with open(pair_path, 'r') as f:
        for line in f.readlines():
            pair = parse_line(line)
            if pair is not None:
                pair_list.append(pair)
                # print(pair)
    #print('#pairs:%d' % len(pair_list))
    # compute feature
    pos_img = []
    neg_img = []
    count = 0
    for pair in pair_list:
        count += 1
        #rel_path1 = '%s/%s_%04d.jpg' % (pair[1], pair[1], int(pair[2]))
        #rel_path2 = '%s/%s_%04d.jpg' % (pair[3], pair[3], int(pair[4]))
        rel_path1 = pair[1]
        rel_path2 = pair[2]
        img_path1 = '%s/%s' % (prefix, rel_path1)
        img_path2 = '%s/%s' % (prefix, rel_path2)
        # skip invalid pairs
        if not os.path.exists(img_path1):
            print(img_path1)
        if not os.path.exists(img_path2):
            print(img_path2)
        if not os.path.exists(img_path1) or not os.path.exists(img_path2):
            continue
        img1 = cv2.imread(img_path1)
        #img1 = cv2.cvtColor(img1, cv2.COLOR_RGB2BGR)
        img2 = cv2.imread(img_path2)
        #img2 = cv2.cvtColor(img2, cv2.COLOR_RGB2BGR)
        #print(img1.shape)
        if pair[0]:
            pos_img.append([img1, img2, rel_path1, rel_path2])
        else:
            neg_img.append([img1, img2, rel_path1, rel_path2])
    return pos_img, neg_img
        
        
def extract_feature_each(extractor, img_list):
    feat_list = []
    n = len(img_list)
    idx = 1
    #bar = tqdm(total=n)
    for pair in img_list:
        img1 = pair[0]
        img2 = pair[1]
        feat1 = extractor.extract(img1)
        feat2 = extractor.extract(img2)
        feat_list.append([feat1, feat2])
        if idx > 1:
            print('{}'.format('\b'*10))
        print('{}/{}'.format(idx, n), end='')
        idx += 1
        #bar.update(1)
    #bar.close()
    return feat_list


def extract_feature_batch(extractor, pair_list, size = 0):
    batch_size = extractor.batch_size
    feat_list = []
    npairs = len(pair_list)
    if size == 0:
        size = npairs*2
    size = min(size, npairs*2)
    npairs_todo = size // 2
    nbatch = (size + batch_size - 1) // batch_size
    #bar = tqdm(total=nbatch)
    for batch in range(nbatch):
        # make a batch
        x_list = []
        for i in range(0, batch_size, 2):
            pairid = (batch * batch_size + i) // 2
            if pairid >= npairs:
                pairid = npairs - 1
            x_list.append(pair_list[pairid][0])
            x_list.append(pair_list[pairid][1])
        #
        x_batch = np.stack(x_list, axis=0)
        feat = extractor.extract(x_batch)
        
        for i in range(0,batch_size,2):
            a = feat[i,:]
            p = feat[i+1,:]
            if len(feat_list) < npairs_todo:
                feat_list.append([a, p])
        #bar.update(1)
    #bar.close()
    return feat_list 

    
def extract_feature(extractor, pair_list, size = 0):
    batch_size = extractor.batch_size
    if batch_size > 1:
        return extract_feature_batch(extractor, pair_list, size)
    return extract_feature_each(extractor, pair_list)
    
    
def crop_pair_list(img_list, imsize):
    """
    crop images
    """
    out_list = []
    h, w, c = img_list[0][0].shape
    x1 = int((w - imsize[0])/2)
    y1 = int((h - imsize[1])/2)
    for pair in img_list:
        img1 = pair[0]
        img2 = pair[1]
        img1 = img1[y1:(y1+imsize[1]),x1:(x1+imsize[0]),:]
        img2 = img2[y1:(y1+imsize[1]),x1:(x1+imsize[0]),:]
        out_list.append([img1, img2])
    #print(img1.shape)
    return out_list
    

def norm_pair_list(img_list):
    """
    norm images
    """
    out_list = []
    for pair in img_list:
        img1 = pair[0]
        img2 = pair[1]
        img1 = ( np.float32(img1) - 127.5 ) / 128
        img2 = ( np.float32(img2) - 127.5 ) / 128
        out_list.append([img1, img2])
    return out_list

    
def load_mxnet_bin(path):
    import mxnet as mx
    bins, issame_list = pickle.load(open(path, 'rb'))
    pos_img = []
    neg_img = []
    for i in range(len(issame_list)):
        _bin = bins[i*2]
        img1 = mx.image.imdecode(_bin).asnumpy()
        _bin = bins[i*2+1]
        img2 = mx.image.imdecode(_bin).asnumpy()
        #print('{} {}'.format(i,img1.shape))
        img1 = cv2.cvtColor(img1, cv2.COLOR_RGB2BGR)
        img2 = cv2.cvtColor(img2, cv2.COLOR_RGB2BGR)
        if issame_list[i]:
          pos_img.append([img1, img2])
        else:
          neg_img.append([img1, img2])
    return pos_img, neg_img


def draw_error_pair(pair, dist):
    h, w, c = pair[0].shape
    canvas = np.zeros((h, w*2, c), dtype=np.uint8)
    cvCopy(pair[0], canvas, (0,0,h,w))
    #print(canvas.shape)
    cvCopy(pair[1], canvas, (0,w,h,w))
    cv2.putText(canvas, '%.3f' % dist, (5, 20), 0, 0.6, (0,255,0), 2)
    return canvas

    
def get_verify_args():
    parser = argparse.ArgumentParser(description='face verification', conflict_handler='resolve')
    parser.add_argument("--type", help="caffe | tensorflow | mxnet")
    parser.add_argument("--test_set", help="lfw | cfp | ytf")
    parser.add_argument("--data",   help="lfw.np or pair.txt")
    parser.add_argument("--prefix", help="data prefix")
    parser.add_argument("--model_path", help= 'specify which model to test ')
    parser.add_argument('--image_size', default="112, 96", help='image size height, width')
    parser.add_argument('--net_depth', default=50, help='resnet depth, default is 50')
    parser.add_argument('--batch_size', type=int, default=64, help='batch size to train network')
    parser.add_argument("--model_name", help= 'specify which model to test \n'
                                              ' centerface\n'
                                              ' sphereface\n'
                                              ' AMSoftmax\n'
                                              ' arcface\n')
    parser.add_argument("--dist_type", default='cosine', help="distance measure ['cosine', 'L2', 'SSD']")
    parser.add_argument("--do_mirror", default=False, help="mirror image and concatinate features")
    parser.add_argument("--do_norm", default=True, help="norm image before feed to nets")
    parser.add_argument("--embed_name", help= 'specify output blob name')
    parser.add_argument("--error_dir", default='', help="directory to save error pairs")
    return parser


def get_extractor(args):
    # model
    if args.type == 'caffe':
        from deeploader.plats.caffe.caffe_model_factory import model_factory
        extractor, args.image_size = model_factory(args.model_name, args.do_mirror)
        print('Testing model\t: %s' % (extractor.weight))
        # do norm
        args.do_norm = True
    elif args.type == 'tensorflow':
        from deeploader.plats.tensorflow.resnet50_extractor import get_extractor
        extractor = get_extractor(args)
        # do norm
        args.do_norm = True
    elif args.type == 'mxnet':
        from deeploader.plats.mxnet.mxnet_extractor import MxnetExtractor
        extractor = MxnetExtractor(args.model_path, args.batch_size, args.image_size, args.embed_name)
    return extractor

    
def do_verify(args, extractor):
    output_dir = '.'
    # parse args   
    image_size = args.image_size
    model_name = args.model_name
    test_set = args.test_set
    dist_type = args.dist_type
    do_mirror = args.do_mirror

    # load images
    data_ext = os.path.splitext(args.data)[1]
    if '.np' == data_ext > 0:
        pos_img, neg_img = pickle.load(open(args.data, 'rb'))
        #pos_img, neg_img = pickle.load(open(lfw_data, 'rb'), encoding='iso-8859-1')
    elif '.txt' == data_ext:
        if args.test_set == 'ytf':
            pos_img, neg_img = load_ytf_pairs(args.data, args.prefix)
        else:
            pos_img, neg_img = load_image_paris(args.data, args.prefix)
    elif '.bin' == data_ext:
        pos_img, neg_img = load_mxnet_bin(args.data)
    else:
        if args.test_set.startswith('cfp'):
            from deeploader.dataset.dataset_cfp import CFPDataset
            pos_list_, neg_list_ = CFPDataset(args.data).get_pairs('FP')
            pos_img = load_image_list(pos_list_)
            neg_img = load_image_list(neg_list_)
    # save input images
    pos_raw = pos_img
    neg_raw = neg_img
    
    # abstract
    print('Dataset  \t: %s (%s,%s)' % (args.test_set, args.data, args.prefix))
    print('Pairs    \t: %d/%d' % (len(pos_img), len(neg_img)))
    print('Testing  \t: %s' % model_name)
    print('Distance \t: %s' % dist_type)
    print('Do mirror\t: {}'.format(do_mirror))
    print('Image size\t: {}'.format(image_size))
    print('Do norm  \t: {}'.format(args.do_norm))
    print('Output   \t: {}'.format(args.error_dir))
    # crop 
    pos_img = crop_pair_list(pos_img, image_size)
    neg_img = crop_pair_list(neg_img, image_size)
    # norm
    if args.do_norm == True:
        print('Norm images')
        pos_img = norm_pair_list(pos_img)
        neg_img = norm_pair_list(neg_img)
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
    # save errors
    if args.error_dir:
        pos_dist, neg_dist = compute_distance(pos_list, neg_list, dist_type)
        h, w, c = pos_raw[0][0].shape
        
        # pos
        target_dir = os.path.join(args.error_dir, 'pos')
        false_neg = 0
        for i in range(len(pos_dist)):
            dist = pos_dist[i][0]
            if dist < threshold:
                continue
            false_neg += 1
            pair = pos_raw[i]
            
            # save
            canvas = draw_error_pair(pair, dist)
            #img_path = target_dir + '/%.3f_%d_%d.jpg' % (dist, i, false_neg)
            img_path = target_dir + '/%d.jpg' % (i)
            makedirs(img_path)
            cv2.imwrite(img_path, canvas)
        # neg
        target_dir = os.path.join(args.error_dir, 'neg')
        false_pos = 0
        for i in range(len(neg_dist)):
            dist = neg_dist[i][0]
            if dist > threshold:
                continue
            false_pos += 1
            pair = neg_raw[i]
            # save
            canvas = draw_error_pair(pair, dist)
            #img_path = target_dir + '/%.3f_%d_%d.jpg' % (dist, i, false_pos)
            img_path = target_dir + '/%d.jpg' % (i)
            makedirs(img_path)
            cv2.imwrite(img_path, canvas)

    return precision, std


if __name__ == '__main__':
    parser = get_verify_args()
    args = parser.parse_args()
    # parse args   
    image_size = args.image_size.split(',')
    args.image_size = (int(image_size[1]), int(image_size[0])) 
    extractor = get_extractor(args)
    do_verify(args, extractor)
   
   

