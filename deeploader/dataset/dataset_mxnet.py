# -*- coding:utf-8 -*-  
from __future__ import print_function
import numpy as np
import os
import sys
import time
import random
import math
import io
import mxnet as mx
import cv2
import numbers
from PIL import Image
import PIL.Image
from deeploader.dataset.dataset_cls import ClassifyDataset


def read_idx(p):
    s = p[0]
    label = p[1]
    header, img_str = mx.recordio.unpack(s)
    label = header.label
    if not isinstance(label, numbers.Number):
        label = label[0]

    img = cv2.imdecode(np.fromstring(img_str, np.uint8), 1)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    return img, label

    # assert label == label_
    # img, label = p[0], p[1]
    # encoded_jpg_io = io.BytesIO(img)
    # image = PIL.Image.open(encoded_jpg_io)
    # # np_img = np.array(image)
    # # img = cv2.cvtColor(np_img, cv2.COLOR_RGB2BGR)
    # img = np.array(image)
    # return img, label


class MxReader(ClassifyDataset):
    def __init__(self, datadir, name='mxrec'):
        ClassifyDataset.__init__(self, name)
        title = 'train'
        if datadir.find('.rec') > 0:
            datadir, fname = os.path.split(datadir)
            title, ext = fname.split('.')
        self.datadir = datadir
        idx_path = os.path.join(datadir, title+'.idx')
        bin_path = os.path.join(datadir, title+'.rec')
        imgrec = mx.recordio.MXIndexedRecordIO(idx_path, bin_path, 'r')
        s = imgrec.read_idx(0)
        header, _ = mx.recordio.unpack(s)
        assert header.flag > 0
        max_index = int(header.label[0])
        min_seq_id = max_index
        max_seq_id = int(header.label[1])
        identities = max_seq_id - min_seq_id
        id2range = [[] for i in range(identities)]
        for id in range(identities):
            identity = id + min_seq_id
            s = imgrec.read_idx(identity)
            header, _ = mx.recordio.unpack(s)
            a, b = int(header.label[0]), int(header.label[1])
            size = b - a
            # extract label
            s = imgrec.read_idx(a)
            header, img_str = mx.recordio.unpack(s)
            label = header.label
            if not isinstance(label, numbers.Number):
                label = label[0]
            label = int(label)
            id2range[label] = [a, b, size]
        # print(id2range)
        self.imgrec = imgrec
        self.images = max_index - 1
        self.identities = identities
        self.id2range = id2range

    def size(self):
        return self.images

    def numOfClass(self):
        return self.identities

    def findLabel(self, index):
        low = 0
        high = len(self.id2range)
        while low <= high:
            mid = int((low + high) / 2)
            y = self.id2range[mid]
            if index >= y[0] and index < y[1]:
                return mid
            elif index >= y[1]:
                low = mid + 1
            else:
                high = mid - 1

    def classCounts(self):
        return [x[2] for x in self.id2range]

    def getData(self, index):
        index = int(index)
        index += 1
        # index to label
        label = -1
        s = self.imgrec.read_idx(index)
        img, label = read_idx((s, label))
        return {'img': img, 'label': label, 'index': index}

    def getClassData(self, classId):
        a, b, _ = self.id2range[classId]
        a -= 1
        b -= 1
        return [i for i in range(a, b)]


if __name__ == '__main__':
    dataset = MxReader('F:/data/nsfw/rec/test.rec')
    dataset.verbose()
    for i in range(dataset.size()):
        data = dataset.getData(i)
        img, label = data['img'], data['label']
        print("%d label:%d [%d, %d]" % (i, label, img.shape[0], img.shape[1]))
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        name = 'POS' if label else 'NEG'
        cv2.imshow(name, img)
        cv2.waitKey(10)