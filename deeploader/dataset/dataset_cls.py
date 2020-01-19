# coding=utf-8
import os
import sys
from deeploader.dataset.dataset_base import ArrayDataset

class ClassifyDataset(ArrayDataset):

    def verbose(self):
        print('Dataset:%8s size:%6d nclass:%d' % (self.name, self.size(), self.numOfClass()))
        if self.numOfClass() < 10:
            for c in range(self.numOfClass()):
                print('%2d: %6d' %(c, len(self.getClassData(c))))

    def numOfClass(self):
        return 0
        
    @property
    def class_nums(self):
        return self.numOfClass()
        
    def getClassData(self, classId):
        return []
        
  