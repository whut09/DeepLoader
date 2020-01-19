# coding=utf-8
import os
import glob
import sys
import random

class DataIter(object):
    def __len__(self):
        return self.size()
        
    def __iter__(self):
        return self
        
    def __next__(self):
        return self.next()
    
    def size(self):
        pass
        
    def next(self):
        pass
        
        
    def reset(self):
        pass
        
    def shuffle(self):
        pass
 

class SeqentialIter(DataIter):
    def __init__(self, dataset):
        self.dataset = dataset
        self.cursor = 0
        self._size = dataset.size()
        
        
    def getDataset(self):
        return self.dataset
        
        
    def size(self):
        return self._size
        
        
    def next(self):
        _data_ = self.dataset.getData(self.cursor)
        self.cursor += 1
        self.cursor %= self._size
        return _data_
        
        
    def reset(self):
        self.cursor = 0 
        

class ShuffleIter(DataIter):
    def __init__(self, dataset):
        self.dataset = dataset
        self._size = dataset.size()
        self.reset()
        
        
    def getDataset(self):
        return self.dataset
        
        
    def size(self):
        return self._size
        
        
    def next(self):
        _data_ = self.dataset.getData(self.ibuf[self.cursor])
        self.cursor += 1
        self.cursor %= self._size
        return _data_
        
        
    def reset(self):
        self.cursor = 0
        self.ibuf = [i for i in range(self._size)]

    def shuffle(self):
        random.shuffle(self.ibuf)
        
        
class BalancedIter(DataIter):
    def __init__(self, dataset):
        self.dataset = dataset
        self._size = dataset.size()
        self._nclass = dataset.numOfClass()
        self.reset()
        
        
    def getDataset(self):
        return self.dataset
        
        
    def size(self):
        return self._size
        
        
    def next(self):
        # 
        class_id = self.cbuf[self.class_idx]
        l = self.dataset.getClassData()
        n = len(l)
        cidx = self.cidx[self.class_idx]
        _data_ = self.dataset.getData(l[cidx])
        # sample idx
        cidx += 1
        cidx %= n
        self.cidx[self.class_idx] = cidx
        # class idx
        self.class_idx += 1
        self.class_idx %= self._nclass
        # cursor
        self.cursor += 1
        return _data_
        
        
    def reset(self):
        self.cursor = 0
        self.class_idx = 0
        self.cbuf = [i for i in range(self._nclass)]
        self.cidx = [0 for i in range(self._nclass)]
        

    def shuffle(self):
        random.shuffle(self.cbuf)
        