#!/usr/bin/python
# -*- coding: UTF-8 -*-
import numpy as np


class Histogram(object):
    def __init__(self, nbins=10, margin=0, nan_policy='easy'):
        '''
        Histogram helper
        :param nbins: num of loss histgram bins
        :param margin: loss range estimation with the middle 1-2*margin percent of all samples
        :param nan_policy: `drop`(never to be used), `easy`(to 1th bin), 'hard'(to the last bin),
                           'rand'(rand bin), followed by accept ratio
        '''
        self.margin = margin
        # parse nan policy
        segs = nan_policy.split('-')
        self.nan_ratio = 1
        self.nan_policy = segs[0]
        if len(segs) == 2:
            self.nan_ratio = float(segs[1])
        self.nbins = nbins

    def _get_bin_idx(self, val):
        idx = int((val - self.val_min) / self.bin_step)
        idx = max(0, idx)
        idx = min(idx, self.nbins - 1)
        return idx

    def update(self, data):
        self.loss = np.array(data)
        self.ds_size = len(data)
        # select samples based on their latest loss values
        # estimate density
        rank = np.argsort(self.loss)
        margin = self.margin
        idx10 = int(self.ds_size * margin)
        idx90 = int(self.ds_size * (1 - margin))
        idx90 = min(idx90, self.ds_size - 1)
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
        print('Data min:%f, max:%f, range:%f, bin_step:%f' % (
        val_min, self.val_max, val_range, self.bin_step))
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
                    val = self.val_min * ((random.random() - 0.5) * 0.1 + 1)
                elif self.nan_policy == 'hard':
                    val = self.val_max * ((random.random() - 0.5) * 0.1 + 1)
                elif self.nan_policy == 'rand':
                    rr = random.random()
                    val = self.val_min * rr + (1 - rr) * self.val_max
                print('take val %f' % val)
                # hack value
                self.loss[i] = val
            else:
                valid_size += 1

            idx = self._get_bin_idx(self.loss[i])
            hist[idx] += 1
            bin_stubs[idx].append(i)
        if valid_size < self.ds_size:
            print('Nan number %d, drop:%d' % (self.ds_size - valid_size, drop_size))
        # normalize hist
        self.hist = hist
        self.norm = []
        for bin in self.hist:
            self.norm.append(float(bin) / valid_size)

    def verbose(self):
        # show hist
        print('Hist:')
        for i in range(self.nbins):
            print('%2d max: %8.3f  hist: %5.2f  count:%d' % (i,
                                                             self.bin_max[i],
                                                             self.norm[i] * 100.0,
                                                             self.hist[i]))

    def __call__(self, x):
        i = 0
        accu = 0
        assert x <= 1
        while accu + self.norm[i] < x and i < len(self.norm):
            accu += self.norm[i]
            i += 1
        left = x - accu
        y = self.val_min + (i + left) * self.bin_step
        return y

    def next(self):
        x = np.random.random()
        return self.__call__(x)
