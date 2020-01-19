
import numpy as np
import sys
import time

from .dataset_list import FileListReader
from .dataset_mxnet import MxReader
from .dataset_dir import FileDirReader
from .dataset_multi import MultiDataset
from .config import Config


def get_WebFace(config, count = -1):
    ds = FileDirReader(config.get('WebFace').datadir, name='WebFace')
    return ds

def get_ms1m(config):
    ds = MxReader(config.get('ms1m').mxrec, name='ms1m')
    return ds

def get_vgg(config):
    ds = MxReader(config.get('vgg').mxrec, name='vgg')
    return ds    

def get_glintasia(config):
    ds = MxReader(config.get('glintasia').mxrec, name='glintasia')
    return ds

def get_glintasia80(config):
    ds = FileDirReader(config.get('glintasia80').datadir, name='glintasia80', dir_as_label=True)
    return ds

def get_emore(config):
    ds = MxReader(config.get('emore').mxrec, name='emore')
    return ds
    
def dataset_factory(name, config):
    dict = {
        'WebFace'   : get_WebFace,
        'ms1m'      : get_ms1m,
        'vgg'       : get_vgg,
        'glintasia' : get_glintasia,
        'glintasia80':get_glintasia80,
        'emore'     : get_emore,
    }
    return dict[name](config)
 
if __name__ == '__main__':
    test_db()
    
        
