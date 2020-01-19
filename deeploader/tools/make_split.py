# -*- coding:utf-8 -*-  
from __future__ import print_function
import random
import argparse
import glob
import os
from deeploader.util.fileutil import makedirs, read_lines, write_lines
     
def get_parser():
    parser = argparse.ArgumentParser(description='parameters test')
    parser.add_argument('--src',      default='.', type=str,   help='source dir')
    parser.add_argument('--splits',   default='0.8/0.1/0.1',type=str,     help='train/val/test')
    parser.add_argument('--dst',      default='.', type=str,    help='list file dir')
    parser.add_argument('--shuffle',  default=1, type=int,     help='list file dir')
    parser.add_argument('--norm',     default=0, type=int,     help='list file dir')
    args = parser.parse_args()
    return args
    

def make_splits(src, splits, dst, shuffle, norm):
    files = glob.glob(src)
    if shuffle:
        random.shuffle(files)
    shuffled = files
    # parse splits
    segs = splits.split('/')
    segs = [float(s) for s in segs]
    # norm ratios
    if norm:
        sum_w = sum(segs)
        ratios = [x/sum_w for x in segs]
    else:
        ratios = segs
    # write lists
    titles = ['train.txt', 'val.txt', 'test.txt']
    start_idx = 0
    total = len(shuffled)
    print('total:{}, splits:{}'.format(total,ratios))
    for idx, r in enumerate(ratios):
        if idx >= 3:
            break
        if r < 0.001:
            continue
            
        n = int(total * r)
        if idx == 2:
            left = total - start_idx
            if left - n < 3:
                n = left
        end_idx = start_idx+n
        seg = shuffled[start_idx:end_idx]
        list_path = os.path.join(dst, titles[idx])
        makedirs(list_path)
        # print(seg)
        write_lines(list_path, seg)
        start_idx = end_idx


if __name__ == '__main__':
    args = get_parser()
    make_splits(args.src, args.splits, args.dst, args.shuffle, args.norm)
    