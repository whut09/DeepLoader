# coding:utf-8
from __future__ import print_function

import os
import sys
import time

class Project(object):
    def __init__(self, projdir):
        self.projdir = projdir
        self.ckptdir = '%s/ckpt' % (self.projdir)
        try:
            os.makedirs(self.ckptdir)
        except:
            pass
            
    @staticmethod
    def make_project(proj):
        timestr = time.strftime('%Y%m%d_%H.%M.%S', time.localtime(time.time()))
        projdir = '%s-%s' % (proj, timestr)
        try:
            os.makedirs(projdir)
        except:
            pass
        return projdir
        
    def proj_dir(self):
        return self.projdir
        
    def log_path(self):
        return os.path.join(self.projdir, 'log.txt')
        
    def tensorboard_dir(self):
        return self.projdir
        
    def ckpt_dir(self):
        return self.ckptdir
    
    def ckpt_path(self, step, epoch=-1):
        if epoch >= 0:
            ckpt = '%s/step-%05d_epoch-%03d' % (self.ckptdir, step, epoch)
            return ckpt
        return '%s/step-%05d' % (self.ckptdir, step)
    
    
    