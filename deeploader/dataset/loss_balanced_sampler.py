# -*- coding:utf-8 -*-  
from __future__ import print_function

import random
import numpy as np
from deeploader.dataset.iter_base import DataIter
from deeploader.util.opencv import *

MAX_LOSS_VAL = 0.1


class LossBalancedSampler(DataIter):
    def __init__(self, dataset, start_epoch=2, nbins=10, margin=0.05,
                 sample_by='round', drop_hard=0, nan_policy='easy',**kargs):
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
        self.dataset = dataset
        self.ds_size = dataset.size()
        self.margin = margin
        self.start_epoch = start_epoch
        self.sample_by = sample_by
        self.drop_hard = drop_hard
        # parse nan policy
        segs = nan_policy.split('-')
        self.nan_ratio = 1
        self.nan_policy = segs[0]
        if len(segs) == 2:
            self.nan_ratio = float(segs[1])
        self.loss = np.zeros(self.ds_size, dtype=np.float32)
        self.rounds = np.zeros(self.ds_size, dtype=np.int32)
        self.loss[:] = MAX_LOSS_VAL
        self.rounds[:] = 0
        self.ibuf = [i for i in range(self.ds_size)]
        self.nbins = nbins
        self.cursor = 0
        self.epoch = -1
        # verbose
        print('LossBalancedSampler start_epoch:%d, margin:%f, drop_hard:%s, nan_policy:%s-%.2f' %
              (start_epoch, margin, drop_hard, self.nan_policy, self.nan_ratio))

    def getDataset(self):
        return self.dataset

    def size(self):
        return len(self.ibuf)

    def next(self):
        index = self.ibuf[self.cursor]
        _data_ = self.dataset.getData(index)
        _data_['index'] = index
        # track freq
        self.rounds[index] += 1
        self.cursor += 1
        self.cursor %= self.size()
        return _data_

    def update_prob(self, indices, losses):
        # print('update prob called')
        self.loss[indices] = losses

    def _get_bin_idx(self, val):
        idx = int((val - self.val_min) / self.bin_step)
        idx = max(0, idx)
        idx = min(idx, self.nbins - 1)
        return idx

    def reset(self):
        self.epoch += 1
        self.cursor = 0
        if self.epoch < self.start_epoch:
            self.ibuf = [i for i in range(self.ds_size)]
            print('Start epoch:%d  size:%d/%d' % (self.epoch, len(self.ibuf), self.ds_size))
            return
        # select samples based on their latest loss values
        # estimate density
        rank = np.argsort(self.loss)
        margin = self.margin
        idx10 = int(self.ds_size * margin)
        idx90 = int(self.ds_size * (1 - margin))
        top10 = self.loss[rank[idx10]]
        top90 = self.loss[rank[idx90]]
        val_range = (top90 - top10) / (1 - 2 * margin)
        val_min = top10 - val_range * margin

        # [0] : x < step[0]
        # [i] : step[i-1] <= x < step[i]
        # [nbins-1]: x >= step[nbins-2]
        self.bin_step = val_range / self.nbins
        self.bin_max = np.zeros(self.nbins)
        
        self.val_min = val_min
        self.val_max = val_min + val_range
        self.bin_max[-1] = self.val_max
        for i in range(self.nbins - 1):
            self.bin_max[i] = val_min + (i + 1) * self.bin_step
        # print(self.loss)
        print('Loss min:%f, max:%f, range:%f, bin_step:%f' % (val_min, self.val_max, val_range, self.bin_step))
        # histogram
        # N.B. init to one
        valid_size = 0
        drop_size = 0
        bin_stubs = [[] for i in range(self.nbins)]
        hist = np.zeros(self.nbins)
        for i in range(self.ds_size):
            val = self.loss[i]
            if val != val:
                # policy: drop
                if self.nan_policy == 'drop':
                    drop_size += 1
                    continue
                # accept with prob
                seed = random.random()
                if seed > self.nan_ratio:
                    drop_size += 1
                    continue
                if self.nan_policy == 'easy':
                    val = self.val_min * ((random.random()-0.5)*0.1 + 1)
                elif self.nan_policy == 'hard':
                    val = self.val_max * ((random.random()-0.5)*0.1 + 1)
                elif self.nan_policy == 'rand':
                    rr = random.random()
                    val = self.val_min * rr + (1-rr) * self.val_max
                print('take val %f' % val)
                # hack value
                self.loss[i] = val
            else:
                valid_size += 1

            idx = self._get_bin_idx(self.loss[i])
            hist[idx] += 1
            bin_stubs[idx].append(i)
        if valid_size < self.ds_size:
            print('Nan number %d, drop:%d' % (self.ds_size-valid_size, drop_size))
        # select sample for all stubs
        ave_samples_per_bin = valid_size // self.nbins
        under_count = 0
        under_total = 0
        under_max = 0
        for i in range(self.nbins):
            binsize = len(bin_stubs[i])
            if binsize <= ave_samples_per_bin:
                under_count += 1
                under_total += binsize
                under_max = max(under_max, binsize)
        # decide ave bin size
        under_ave = under_total/under_count
        ave_samples_per_bin = int(min(under_max, under_ave*2))

        self.ibuf = []
        bin_sample_size = []
        for i in range(self.nbins):
            stub_list = bin_stubs[i]
            samples_per_bin = ave_samples_per_bin
            sample_by = self.sample_by
            # take care of drop hard
            if i == self.nbins - 1 and self.drop_hard:
                need = min(len(stub_list), ave_samples_per_bin)
                samples_per_bin = int(need * (1 - self.drop_hard))
                sample_by = 'easy'

            if len(stub_list) <= samples_per_bin:
                self.ibuf += stub_list
                bin_sample_size.append(len(stub_list))
                continue

            # sampling
            bin_sample_size.append(samples_per_bin)
            if sample_by == 'round':
                # select least trained samples
                r_list = self.rounds[stub_list]
                idx = np.argsort(r_list)
                selected = idx[:samples_per_bin]
                # print(selected)
                _stub_list = np.array(stub_list)
                selected = _stub_list[selected]
                selected = selected.tolist()
            elif sample_by == 'rand':
                # random select
                random.shuffle(stub_list)
                selected = stub_list[:samples_per_bin]
            elif sample_by == 'easy':
                # select least trained samples
                r_list = self.loss[stub_list]
                idx = np.argsort(r_list)
                selected = idx[:samples_per_bin]
                # print(selected)
                _stub_list = np.array(stub_list)
                selected = _stub_list[selected]
                selected = selected.tolist()
            else:
                selected = stub_list[:samples_per_bin]

            self.ibuf += selected
        # show hist
        print('Loss hist:')
        for i in range(self.nbins):
            print('%2d max: %8.3f  hist: %5.2f  r:%5.2f' % (i,
                                                            self.bin_max[i],
                                                            hist[i] * 100.0 / self.ds_size,
                                                            100. * bin_sample_size[i] / len(
                                                                self.ibuf)))

        print('Start epoch:%d  size:%d/%d' % (self.epoch, len(self.ibuf), self.ds_size))

    def shuffle(self):
        random.shuffle(self.ibuf)


if __name__ == '__main__':
    pass
