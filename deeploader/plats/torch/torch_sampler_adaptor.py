# -*- coding:utf-8 -*-  
from __future__ import print_function

import random
import numpy as np
from deeploader.dataset.iter_base import DataIter
from torch.utils.data.sampler import Sampler
from deeploader.dataset.loss_balanced_sampler import LossBalancedSampler


class LossBalancedSamplerAdaptor(Sampler, LossBalancedSampler):
    def __init__(self, dataset, *args, **kargs):
        '''
        Select training samples for each epoch according to their loss values.
        :param dataset: input dataset
        :param start_epoch: num of epochs to start working
        :param nbins: num of loss histgram bins
        :param margin: loss range estimation with the middle 1-2*margin percent of all samples
        :param sample_by: sample rule in each bin if it's more than average: `round`(least trained), `easy`(minimal loss), 'rand'(random)
        :param drop_hard: percentage to drop for samples in the hardest bin(with largest loss value)
        :param nan_policy: `drop`(never to be used), `easy`(to 1th bin), 'hard'(to the last bin),
                           'rand'(rand bin), followed by accept ratio
        '''
        LossBalancedSampler.__init__(self, dataset, *args, **kargs)
        self.do_shuffle = kargs.get('shuffle', True)

    def __iter__(self):
        self.reset()
        if self.do_shuffle:
            self.shuffle()
        return iter(self.ibuf)

    def __len__(self):
        return len(self.ibuf)
