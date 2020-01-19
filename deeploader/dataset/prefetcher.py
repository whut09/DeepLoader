# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

"""Data iterators for common data formats."""
from __future__ import absolute_import
import threading

import numpy as np


def _apply_transform(_transforms, data):
    if not _transforms:
        return data
    if isinstance(_transforms, dict):
        for k, v in _transforms.items():
            if k in data:
                data[k] = v(data[k])
    else:
        data = _transforms(data)
    return data


def get_batch_dict(batch, trans=None):
    """
    :param batch:list of dict
    :param trans: a dict of {'key':xxx,'fun':yyy} to transform array to tensor(respect to framework)
    :return:a dict of array or tensor(if trans not None)
    """
    if isinstance(batch, list):
        keys = batch[0].keys()
        key_set = set(keys)
        for i in range(1, len(batch)):
            sub = set(batch[i].keys())
            key_set = key_set.intersection(sub)
        keys = list(key_set)
        batch_dict = {}
        for k in keys:
            batch_dict[k] = []
        for item in batch:
            for k in keys:
                batch_dict[k].append(item[k])
        for k in keys:
            if not isinstance(batch_dict[k][0], str):
                batch_dict[k] = np.array(batch_dict[k])
        if trans is not None:
            for k in trans:
                func = trans[k]
                batch_dict[k] = func(batch_dict[k])
        return batch_dict
    else:
        raise TypeError("batch must be a list of dictionary")

        
class PrefetchIter(object):
    def __init__(self, iter, batch_size, shuffle=True, transforms=None):
        self.iter = iter
        self.shuffle = shuffle
        self._batch_size = batch_size
        self._transforms = transforms
        self._batch_per_epoch = int((iter.size() + batch_size - 1) / batch_size)
        self.data_ready = threading.Event()
        self.data_taken = threading.Event()
        self.data_taken.set()
        self.batch_index = 0
        if self.shuffle:
            self.iter.shuffle()
            
        self.started = True
        self.current_batch = None
        self.next_batch = [None for i in range(batch_size)]

        def prefetch_func(self):
            """Thread entry"""
            while True:
                self.data_taken.wait()
                if not self.started:
                    break
                # load a batch
                for i in range(self._batch_size):
                    self.next_batch[i] = self.iter.next()
                    # apply transform
                    self.next_batch[i] = self.apply_transform(self.next_batch[i])

                self.data_taken.clear()
                self.data_ready.set()
        self.prefetch_threads = threading.Thread(target=prefetch_func, args=[self]) 
        self.prefetch_threads.setDaemon(True)
        self.prefetch_threads.start()

    def apply_transform(self, data):
        return _apply_transform(self._transforms, data)

    @property
    def batch_per_epoch(self):
        return self._batch_per_epoch
        
    @property    
    def batch_size(self):
        return self._batch_size
    
    @property
    def size(self):
        return self.iter.size()

    def __len__(self):
        return self._batch_per_epoch

    def __iter__(self):
        return self
 
    def __next__(self):
        return self.next()

    def verbose(self):
        print('TotalSize :%d' % self.iter.size())
        print('BatchSize :%d' % self._batch_size)
        print('BatchEpoch:%d' % self._batch_per_epoch)

    def __del__(self):
        self.started = False
        self.data_taken.set()
        self.prefetch_threads.join()

    def reset(self):
        self.batch_index = 0
        self.data_ready.wait()
        self.iter.reset()
        if self.shuffle:
            self.iter.shuffle()
        self.data_ready.clear()
        self.data_taken.set()

    def iter_next(self):
        self.data_ready.wait()
        if self.next_batch[0] is None:
            for i in self.next_batch:
                assert i is None, "Number of entry mismatches between iterators"
            return False
        else:
            self.current_batch = self.next_batch
            self.next_batch = [None for i in range(self._batch_size)]
            self.data_ready.clear()
            self.data_taken.set()
            return True

    def batch_data(self):
        return get_batch_dict(self.current_batch)

    def next(self):
        if self.batch_index >= self._batch_per_epoch:
            raise StopIteration
        self.batch_index += 1
        self.iter_next()
        return get_batch_dict(self.current_batch)


class BatchIter(object):
    def __init__(self, iter, batch_size, shuffle=True, transforms=None):
        self.iter = iter
        self.shuffle = shuffle
        self._batch_size = batch_size
        self._transforms = transforms
        self._batch_per_epoch = int((iter.size() + batch_size - 1) / batch_size)
        self.batch_index = 0
        if self.shuffle:
            self.iter.shuffle()

        self.current_batch = None
        self.next_batch = [None for i in range(batch_size)]

    def prefetch_func(self):
        # load a batch
        for i in range(self._batch_size):
            self.next_batch[i] = self.iter.next()
            # apply transform
            self.next_batch[i] = self.apply_transform(self.next_batch[i])

    def apply_transform(self, data):
        return _apply_transform(self._transforms, data)

    @property
    def batch_per_epoch(self):
        return self._batch_per_epoch

    @property
    def batch_size(self):
        return self._batch_size

    @property
    def size(self):
        return self.iter.size()

    def __len__(self):
        return self._batch_per_epoch

    def __iter__(self):
        return self

    def __next__(self):
        return self.next()

    def verbose(self):
        print('TotalSize :%d' % self.iter.size())
        print('BatchSize :%d' % self._batch_size)
        print('BatchEpoch:%d' % self._batch_per_epoch)

    def __del__(self):
        pass

    def reset(self):
        self.batch_index = 0
        self.iter.reset()
        if self.shuffle:
            self.iter.shuffle()
        # update size
        self._batch_per_epoch = int((self.iter.size() + self.batch_size - 1) / self.batch_size)

    def iter_next(self):
        self.prefetch_func()
        self.current_batch = self.next_batch
        self.next_batch = [None for i in range(self._batch_size)]
        return True

    def batch_data(self):
        return get_batch_dict(self.current_batch)

    def next(self):
        if self.batch_index >= self._batch_per_epoch:
            raise StopIteration
        self.batch_index += 1
        self.iter_next()
        return get_batch_dict(self.current_batch)