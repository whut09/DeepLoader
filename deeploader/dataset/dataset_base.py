# coding=utf-8
class ArrayDataset(object):
    def __init__(self, name='--'):
        self.name = name
        
    def verbose(self):
        print('Dataset:%s size:%d' % (self.name, self.size()))
      
    def size(self):
        return 0
        
    def getData(self, index):
        return {}
        
    def __getitem__(self, index):
        return self.getData(index)

    def __len__(self):
        return self.size()
        
  