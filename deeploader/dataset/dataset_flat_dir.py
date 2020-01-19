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
from deeploader.dataset.dataset_base import ArrayDataset


def allowed_file(filename, exts=None):
    ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'gif', 'bmp', 'tif'])
    if exts == None:
        exts = ALLOWED_EXTENSIONS

    return '.' in filename and filename.rsplit('.', 1)[1].lower() in exts
    

class FlatDirDataset(ArrayDataset):
    def __init__(self, datadir, name='--', exts=None):
        ArrayDataset.__init__(self, name)
        self.datadir = datadir
        # scan files
        filenames = os.listdir(datadir)
        filenames.sort()
        self.list = []
        
        for filename in filenames:
            if not allowed_file(filename, exts):
                continue
            self.list.append(filename)
 
    def size(self):
        return len(self.list)
  
    def getData(self, index):
        rel_path = self.list[index]
        path = os.path.join(self.datadir, rel_path)
        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        
        return {'img':img, 'path': rel_path }
        
     
    