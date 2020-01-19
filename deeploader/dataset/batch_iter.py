from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from deeploader.dataset.iter_base import SeqentialIter, ShuffleIter
from deeploader.dataset.prefetcher import BatchIter


class DeepDataIter(object):
    def __init__(self, dataset, batch_size, data_shape, sampler=None, shuffle=True, transforms=None,
                 data_name='data',
                 label_name='softmax_label'):
        self.data_shape = (batch_size,) + data_shape
        # print(self.data_shape)
        self.label_shape = (batch_size,)
        self.provide_data = [(data_name, self.data_shape)]
        self.batch_size = batch_size
        self.image_size = '%d,%d' % (data_shape[1], data_shape[2])
        self.provide_label = [(label_name, (batch_size,))]
        self.dataset = dataset
        if sampler is None:
            if shuffle:
                self.iter = BatchIter(ShuffleIter(dataset), batch_size, shuffle,
                                      transforms=transforms)
            else:
                self.iter = BatchIter(SeqentialIter(dataset), batch_size, shuffle,
                                      transforms=transforms)
        else:
            self.iter = BatchIter(sampler, batch_size, shuffle, transforms=transforms)

        self.num_classes = dataset.numOfClass()
        self.iter.reset()

    @property
    def batch_per_epoch(self):
        return self.iter.batch_per_epoch

    def __len__(self):
        return self.batch_per_epoch

    def name(self):
        return self.dataset.name

    def reset(self):
        self.iter.reset()

    def batch_data(self):
        return self.iter.batch_data()

    def update_prob(self, indices, losses):
        if hasattr(self.iter.iter, 'update_prob'):
            self.iter.iter.update_prob(indices, losses)

    def next(self):
        data = self.iter.next()
        return data
