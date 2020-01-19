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
import sys
sys.path.append(".")

from util.fileutil import makedirs

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
    
    
def save_image_pairs(pair_path, prefix, dst):
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
    bins = []
    issame_list = []
    count = 0
    p_list = []
    n_list = []
    for idx, pair in enumerate(pair_list):
        if pair[0]:
            p_list.append(idx)
        else:
            n_list.append(idx)
    
    sets = [p_list, n_list]
    for idx in range(len(p_list)):
        for s in sets:
            pair = pair_list[s[idx]]
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
            with open(img_path1, 'rb') as f:
                img1 = f.read()
            with open(img_path2, 'rb') as f:
                img2 = f.read()
            bins.append(img1)
            bins.append(img2)
            if pair[0]:
                issame_list.append(True)
            else:
                issame_list.append(False)
    with open(dst, 'wb') as f:
        pickle.dump((bins, issame_list), f)

        
def save_cfp_pairs(data, dst):
    from dataset.dataset_cfp import CFPDataset
    pos_list_, neg_list_ = CFPDataset(data).get_pairs('FP')
    sets = [pos_list_, neg_list_]
    
    bins = []
    issame_list = []
    for idx in range(len(pos_list_)):
        for sid, s in enumerate(sets):
            pair = s[idx]
            img_path1 = pair[0]
            img_path2 = pair[1]
            # skip invalid pairs
            if not os.path.exists(img_path1):
                print(img_path1)
            if not os.path.exists(img_path2):
                print(img_path2)
            if not os.path.exists(img_path1) or not os.path.exists(img_path2):
                continue
            with open(img_path1, 'rb') as f:
                img1 = f.read()
            with open(img_path2, 'rb') as f:
                img2 = f.read()
            bins.append(img1)
            bins.append(img2)
            if sid == 0:
                issame_list.append(True)
            else:
                issame_list.append(False)
    with open(dst, 'wb') as f:
        pickle.dump((bins, issame_list), f)
            
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--type", help="lfw | cfp")
    parser.add_argument("--data",   help="lfw.np or pair.txt")
    parser.add_argument("--prefix", help="data prefix")
    parser.add_argument("--dst", help="dst bin path")
    args = parser.parse_args()

    # load images
    if args.type == 'lfw':
        save_image_pairs(args.data, args.prefix, args.dst)
    elif args.type == 'cfp':
        save_cfp_pairs(args.data, args.dst) 
    
   
   

