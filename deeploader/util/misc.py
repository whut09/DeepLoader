# -*- coding:utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import sys


def print_args(args, tag=''):
    arg_dict = vars(args)
    print('-' * 80)
    title = '%s args' % tag
    space = int((80 - len(title)) / 2)
    print(' ' * space, end='')
    print(title)
    print('-' * 80)
    keys = arg_dict.keys()
    keys.sort()
    for k in keys:
        v = arg_dict[k]
        print('{:16s}: {}'.format(k, v))
    print('-' * 80)
    print('\n', end='')


def merge_args(obj, dst):
    for k, v in obj.__dict__.items():
        if not hasattr(dst, k):
            setattr(dst, k, v)
            # print('copy {} {}'.format(k, v))


def parse_gpus(gpus):
    if isinstance(gpus, str):
        segs = gpus.split(',')
        l = []
        for seg in segs:
            l.append(int(seg.strip()))
        gpus = l
    return gpus


def set_gpus(gpus):
    if isinstance(gpus, str):
        gpu_list = gpus
    else:
        gpu_list = ''
        for i, gpu_id in enumerate(gpus):
            gpu_list += str(gpu_id)
            if i != len(gpus) - 1:
                gpu_list += ','
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu_list


def trim_float(x, n=2):
    fmt = '%%.%df' % n
    x = float(fmt % x)
    return x


def extract_field(obj_list, name):
    ret = []
    for obj in obj_list:
        if name in obj:
            ret.append(obj[name])
    return ret


def write_csv_file(obj_list, csv_path):
    keys = list(obj_list[0].keys())
    keys.sort()
    output_keys = []
    for key in keys:
        if key == 'name':
            output_keys.insert(0, key)
        else:
            output_keys.append(key)
    with open(csv_path, 'w') as f:
        # write header
        for idx, key in enumerate(output_keys):
            if idx > 0:
                f.write(',')
            f.write(key)
        f.write('\n')
        # write items
        for line in obj_list:
            for idx, key in enumerate(output_keys):
                if idx > 0:
                    f.write(',')
                f.write('{}'.format(line[key]))
            f.write('\n')


def get_mod(mod_path):
    mod_dir, mod_name = os.path.split(mod_path)
    mod_name, _ = os.path.splitext(mod_name)
    sys.path.append(mod_dir)
    exp = '__import__(\"%s\")' % mod_name
    mod = eval(exp)
    return mod
