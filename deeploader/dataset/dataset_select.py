import numpy as np
import time
import random
import pickle
from deeploader.dataset.dataset_cls import ClassifyDataset


class DatasetSelect(ClassifyDataset):
    def __init__(self, dataset, num_class=100, max_per_class=-1):
        name = dataset.name + '-filt'
        ClassifyDataset.__init__(self, name)
        self.dataset = dataset
        self.num_class = min(num_class, self.dataset.numOfClass())
        self.dict = {}
        self.list = []
        for i in range(self.num_class):
            class_set = self.dataset.getClassData(i)
            if max_per_class > 0 and len(class_set) > max_per_class:
                random.shuffle(class_set)
                class_set = class_set[:max_per_class]
            start_idx = len(self.list)
            size = len(class_set)
            self.dict[i] = [ start_idx + j for j in range(size) ]
            self.list = self.list + class_set

    def size(self):
        return len(self.list)

    def numOfClass(self):
        return len(self.dict)

    def getData(self, index):
        idx = self.list[index]
        return self.dataset.getData(idx)

    def getClassData(self, classId):
        return self.dict[classId]
