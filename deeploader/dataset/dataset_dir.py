# -*- coding:utf-8 -*-  
from __future__ import print_function
import scipy
import numpy as np
import os
import sys
import time
import random
import math
import cv2
from tqdm import tqdm 
from deeploader.dataset.dataset_cls import ClassifyDataset


def allowed_file(filename, exts=None):
    ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'gif', 'bmp', 'tif'])
    if exts == None:
        exts = ALLOWED_EXTENSIONS

    return '.' in filename and filename.rsplit('.', 1)[1].lower() in exts
    
    
class FileDirReader(ClassifyDataset):
    def __init__(self, datadir, name='--', dir_as_label=False, exts=None):
        ClassifyDataset.__init__(self, name)
        self.datadir = datadir
        # scan files
        if not dir_as_label:
            subdirs = os.listdir(datadir)
            subdirs.sort()
        else:
            subdirs = os.listdir(datadir)
            subdirs = [str(i) for i in range(len(subdirs))]
        label = 0
        index = 0
        self.list = []
        self.dict = {}
        bar = tqdm(total=len(subdirs))
        for subdir in subdirs:
            fullpath = os.path.join(datadir,subdir)
            bar.update(1)
            if not os.path.isdir(fullpath):
                continue
            filenames = os.listdir(fullpath)
            filenames.sort()
            count = 0
            sub_list = []
            for filename in filenames:
                if not allowed_file(filename, exts):
                    continue
                filepath = os.path.join(subdir, filename)
                self.list.append((filepath, label))
                sub_list.append(index)
                index += 1
            if sub_list:
                self.dict[label] = sub_list
                label += 1
        bar.close()

        
    def size(self):
        return len(self.list)

    def numOfClass(self):
        return len(self.dict.keys())

        
    def getData(self, index):
        item = self.list[index]
        rel_path = item[0]
        path = os.path.join(self.datadir, rel_path)
        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        
        return {'img':img, 'label':item[1], 'path': rel_path }
        
    def getClassData(self, classId):
        return self.dict[classId]
     
    