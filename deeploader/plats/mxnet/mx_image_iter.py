from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from mxnet import io
from mxnet import ndarray as nd

from deeploader.dataset.batch_iter import DeepDataIter


class MxDataIter(DeepDataIter, io.DataIter):
    def __init__(self, dataset, batch_size, data_shape, *args, **kargs):
        DeepDataIter.__init__(self, dataset, batch_size, data_shape, *args, **kargs)
        io.DataIter.__init__(self, batch_size)

    def next(self):
        data = self.iter.next()
        img = data['img']
        # print(img)
        img = img.transpose(0, 3, 1, 2)
        batch_data = nd.array(img, dtype=np.uint8).astype(np.float32)
        # batch_data = nd.array(img)
        # print(batch_data.shape)
        # batch_data = nd.transpose(batch_data, axes=(0, 3, 1, 2))
        batch_label = nd.array(data['label'])  # .astype(np.int32)
        batch_index = None
        if 'index' in data:
            batch_index = nd.array(data['index'])
        # print('batch dtype {}'.format(batch_label.dtype))
        return io.DataBatch([batch_data], [batch_label], index=batch_index)
