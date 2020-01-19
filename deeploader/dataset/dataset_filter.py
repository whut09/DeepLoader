import numpy as np
import time
import random
import pickle
from deeploader.dataset.dataset_cls import ClassifyDataset

class DatasetFilter(ClassifyDataset):
    def __init__(self, dataset, filter_pkl):
        name = dataset.name+'-filt'
        ClassifyDataset.__init__(self, name)
        self.dataset = dataset
        with open(filter_pkl, 'rb') as f:
            filter = pickle.load(f)
        #print(filter)
        celeb = list(filter.keys())
        celeb.sort()
        self.filter = filter
        self.celeb = celeb
        self.list = []
        self.dict = {}
        
        for idx in range(len(self.celeb)):
            key = self.celeb[idx]
            v = self.filter[key]
            #print('%d: size:%d' % (label, len(v)))
            n = len(v)
            offset = len(self.list)
            self.dict[idx] = [offset+i for i in range(n)]
            for x in v:
                self.list.append((x, idx))
            
    def size(self):
        return len(self.list)

    def numOfClass(self):
        return len(self.celeb)
    
    def getData(self, index):
        sid, label = self.list[index]
        d = self.dataset.getData(sid)
        d['label'] = label
        return d

    def getClassData(self, classId):
        return self.dict[classId]
        

if __name__ == '__main__':
    from deeploader.dataset.dataset_mxnet import MxReader
    from deeploader.dataset.dataset_dir import FileDirReader
    ms1m = MxReader('faces_ms1m_112x112')
    ms1m.verbose()
    dataset = DatasetFilter(ms1m, 'filter.pkl')
    dataset.verbose()
    #dataset.shuffle()
    for i in range(10000):
        func, param = dataset.nextTask()
        img, label = func(param)
        #print("%d label:%6d [%d, %d]"%(i, label, img.shape[0], img.shape[1]))
        
