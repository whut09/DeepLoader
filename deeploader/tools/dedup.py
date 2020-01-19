#!/usr/bin/env python
# -*- coding:utf-8 -*-

import argparse
import hashlib
import hnswlib
import os
import shutil
import sys
from glob import glob

import numpy as np
from deeploader.util.fileutil import *
from deeploader.util.hashing import MHash
from tqdm import tqdm


def merge_file_list(file_list, dst_dir):
    hash_set = set()
    # file_list = glob(root_dir + '/*/*')
    # print(file_list)
    makedir(dst_dir)
    idx = 0
    for path in file_list:
        # ext type
        if not allowed_file(path):
            continue
        # file size
        fsize = get_file_size(path)
        if fsize < 10000:
            continue
        # duplicate
        hash = get_file_md5(path)
        if hash in hash_set:
            continue
        _, ext = os.path.splitext(path)
        ext = ext.lower()
        dst_path = '%d%s' % (idx, ext)
        dst_path = os.path.join(dst_dir, dst_path)
        # copy file
        shutil.copyfile(path, dst_path)
        hash_set.add(hash)
        idx += 1


def move_file_list(duplicates, dup_dir, image_dir=''):
    makedir(dup_dir)
    for f in duplicates:
        try:
            shutil.move(os.path.join(image_dir, f), dup_dir)
        except:
            pass


def _get_image_feats(image_dir):
    import pickle
    images, feats = None, None
    # check cache
    _, fname = os.path.split(image_dir)
    cache_path = fname + ".pkl"
    if os.path.exists(cache_path):
        with open(cache_path, "rb") as f:
            images, feats = pickle.load(f)
        return images, feats
    # compute
    phash = MHash()
    images, feats = phash.get_image_feats(image_dir)
    with open(cache_path, "wb") as f:
        pickle.dump((images, feats), f)
    return images, feats


def _find_deplicates(feats, min_distance_threshold=10, max_dups=5):
    num_elements, dim = feats.shape
    data_labels = np.arange(num_elements)
    # Declaring index
    p = hnswlib.Index(space='l2', dim=dim)  # possible options are l2, cosine or ip

    # Initing index - the maximum number of elements should be known beforehand
    p.init_index(max_elements=num_elements, ef_construction=200, M=16)
    # Element insertion (can be called several times):
    p.add_items(feats, data_labels)

    # Controlling the recall by setting ef:
    ef = max(num_elements * 0.001, max_dups * 10)
    ef = min(ef, 1000)
    p.set_ef(int(ef))  # ef should always be > k
    K = max_dups + 1
    # Query dataset, k - number of closest elements (returns 2 numpy arrays)
    print('Query knn')
    labels, distances = p.knn_query(feats, k=K)
    # print(labels, distances)
    # print(distances)
    print('Merge result')
    # dist_threshold = dim * (1-min_similarity_threshold)
    dist_threshold = min_distance_threshold
    files_to_remove = set()
    key_set = set()
    dup_dict = {}
    bar = tqdm(total=num_elements)
    for i in range(num_elements):
        if i not in key_set and i not in files_to_remove:
            key_set.add(i)
        # ignore top 1
        for j in range(K):
            k = labels[i, j]
            dist = distances[i, j]
            # ignore self
            if labels[i, j] == i:
                continue
            # check dist
            if dist > dist_threshold:
                break
            if i not in dup_dict:
                dup_dict[i] = [[k, dist]]
            else:
                dup_dict[i].append([k, dist])
            # add
            if k not in key_set and k not in files_to_remove:
                files_to_remove.add(k)
        bar.update(1)
    bar.close()
    return files_to_remove, dup_dict


def find_deplicates_to_remove(image_dir, min_distance_threshold=0.15):
    images, feats = _get_image_feats(image_dir)
    # print(images)
    files_to_remove_ids, dup_dict_ids = _find_deplicates(feats, min_distance_threshold)
    # translate to path
    files_to_remove = [images[i] for i in files_to_remove_ids]
    dup_dict = {}
    for key in dup_dict_ids:
        arr = dup_dict_ids[key]
        path = images[key]
        for idx in range(len(arr)):
            arr[idx][0] = images[arr[idx][0]]
        dup_dict[path] = arr
    return files_to_remove, dup_dict


def plot_duplicates(image_dir, dup_dict):
    import cv2
    for item in dup_dict:
        src_img = cv2.imread(item)
        cv2.imshow('src', src_img)
        for idx, val in enumerate(dup_dict[item]):
            if not os.path.exists(val[0]):
                continue
            dst_img = cv2.imread(val[0])
            txt = '%.3f' % val[1]
            cv2.putText(dst_img, txt, (10, 20), 1, 2, (0, 255, 0), 2)
            cv2.imshow('top-%d' % (idx + 1), dst_img)
        key = cv2.waitKey()
        cv2.destroyAllWindows()
        if key == 27:
            break


def get_parser():
    parser = argparse.ArgumentParser(description='image duplicates removal')
    parser.add_argument('--src', type=str, default='.', help='source dir')
    parser.add_argument('--dst', type=str, default='dup', help='duplicate file dir')
    parser.add_argument('--viz', type=int, default=0, help='show result')
    parser.add_argument('--t', type=float, default=0.2, help='dist threshold')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = get_parser()
    files_to_remove, dup_dict = find_deplicates_to_remove(args.src, args.t)
    if not args.viz:
        move_file_list(files_to_remove, args.dst)
    else:
        plot_duplicates(args.src, dup_dict)
    sys.exit(0)
