import numpy as np
import time
import random
from deeploader.dataset.dataset_cls import ClassifyDataset

class MultiDataset(ClassifyDataset):
    def __init__(self):
        self.datasets = []
        self.metas = []
        self.totalClass = 0
        self.totalSize = 0
        
    def add(self, dataset, id_offset=-1):
        self.datasets.append(dataset)
        # classify data
        num_class = 0
        if hasattr(dataset, 'numOfClass'):
            if id_offset >= 0:
                id_offset = id_offset
                self.totalClass = max(id_offset+dataset.numOfClass(), self.totalClass)
            else:
                id_offset = self.totalClass
                self.totalClass += dataset.numOfClass()
            num_class = dataset.numOfClass()
        meta = {'id_offset': id_offset, 'numOfClass': num_class,
          'index_offset':self.totalSize, 'size': dataset.size()} 
        self.metas.append(meta)

        self._normWeights()
        self.totalSize += dataset.size()
        
    def _normWeights(self):
        sw = 0
        for m in self.metas:
            sw += m['size']
        for m in self.metas:
            m['weight'] = float(m['size'])/sw
        
        
    def size(self):
        return self.totalSize

    def numOfClass(self):
        return self.totalClass
    
    def verbose(self):
        for i in range(len(self.datasets)):
            dataset = self.datasets[i]
            if hasattr(dataset, 'numOfClass'):
                print('Dataset:%16s size:%7d nclass:%6d  %.3f' % (dataset.name, 
                  dataset.size(), dataset.numOfClass(), self.metas[i]['weight']))
            else:
                print('Dataset:%16s size:%7d  %.3f' % (dataset.name, 
                  dataset.size(), self.metas[i]['weight']))

        print('------------------------------------------------------------')
        if hasattr(self.datasets[0], 'numOfClass'):
            print('Dataset:%8s size:%7d nclass:%6d' % ('total', self.size(), self.numOfClass()))
            if self.numOfClass() < 10:
                for c in range(self.numOfClass()):
                    print('%2d: %6d' % (c, len(self.getClassData(c))))
        else:
            print('Dataset:%8s size:%7d' % ('total', self.size()))


    def getData(self, index):
        pos = 0
        for i, ds in enumerate(self.datasets):
            # 
            if index < pos+ds.size():
                _data_ = ds.getData(index - pos)
                if 'label' in _data_:
                    _data_['label'] += self.metas[i]['id_offset']
                return _data_
            pos += ds.size()
        
        
    def getClassData(self, classId):
        l = []
        pos = 0
        for i, m in enumerate(self.metas):
            id_start = m['id_offset']
            id_end = id_start + m['numOfClass']
            index_offset = m['index_offset']
            if classId >= id_start and classId < id_end:
                _data_ = self.datasets[i].getClassData(classId - id_start)
                _data_ = [int(i+index_offset) for i in _data_]
                l = l + _data_
        return l
        
        
if __name__ == '__main__':
    '''
    y = [1,2,3,0]
    y_train = np_utils.to_categorical(y, 4)
    print(y_train)
    exit()
    '''
    dg = DataGeneratorMT(32, 1)
    print(dg.numOfClass())
    for i in range(100000):
        print('batch:%d'%(i))
        start = time.time()
        x, y = dg.getBatch()
        end = time.time()
        #plot_fbank_cm(fbank_feat)
        #print("x.shape:{0}, y.shape:{1}".format(x.shape, y.shape))
        print(y)
        #print('t:%f' % (end - start) )
    dg.close()
        
