import time
from deeploader.dataset.dataset_dir import FileDirReader
from deeploader.dataset.iter_base import ShuffleIter
from deeploader.dataset.prefetcher import PrefetchIter, get_batch_dict


def build_train_dataset_glintasia():
    from dataset.config import Config
    config = Config('./dataset/config-59.ini')
    dataset = MultiDataset()
    
    # glintasia
    ds = dataset_factory('glintasia', config)
    #ds = DatasetFilter(ds, config.get('glintasia').filter)
    dataset.add(ds, id_offset = 0)
    # glintasia80
    ds = dataset_factory('glintasia80', config)
    #ds = DatasetFilter(ds, config.get('glintasia').filter)
    dataset.add(ds, id_offset = 0)
    dataset.verbose()
    train_iter = ShuffleIter(dataset)
    
    return train_iter

iter = build_train_dataset_glintasia()
train_iter = PrefetchIter(iter, batch_size=100)
train_iter.verbose()
for i in range(train_iter.batch_per_epoch):
    s = time.time()
    data = train_iter.next()
    e = time.time()
    print('%d %.1fms' % (i, (e-s) * 1000))
    time.sleep(0.05)
    #data = get_batch_dict(data)
    #print(data)