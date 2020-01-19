# coding=utf-8
import os
import glob
import sys

def readlines(path):
    lines = []
    with open(path, 'r') as f:
        for line in f.readlines():
            lines.append(line.strip())
    return lines
    

def load_fiducial(path):
    lines = readlines(path)
    points = []
    for line in lines:
        x, y = line.split(',')
        x = float(x)
        y = float(y)
        points.append(x)
        points.append(y)
    return points

    
class CFPDataset:
    def __init__(self, rootdir, aligned_dir='Data/112x112'):
        self.rootdir = rootdir
        self.fp_path = os.path.join(rootdir, 'Protocol/Pair_list_P.txt')
        self.ff_path = os.path.join(rootdir, 'Protocol/Pair_list_F.txt')
        self.img_dir = os.path.join(rootdir, 'Protocol')
        # image list
        self.fp_list = self._parse_img_list(self.fp_path)
        self.ff_list = self._parse_img_list(self.ff_path)
        self.aligned_dir = os.path.join(rootdir, aligned_dir)
        # pairs list
        self.fp_pairs = self._scan_splits('FP')
        self.ff_pairs = self._scan_splits('FF')
        
        
    def _scan_splits(self, proto='FP'):
        set_a = self.ff_list
        set_b = self.fp_list
        if proto == 'FF':
            set_b = self.ff_list
        dir = self.img_dir + '/Split/' + proto
        subs = os.listdir(dir)
        same_list = []
        diff_list = []
        for sub in subs:
            sub_dir = os.path.join(dir, sub)
            # FF
            lines = readlines(os.path.join(sub_dir, 'same.txt'))
            for line in lines:
                a, b = line.split(',')
                # to zero-based index
                a = int(a) - 1
                b = int(b) - 1
                same_list.append([set_a[a], set_b[b]])
            # FP
            lines = readlines(os.path.join(sub_dir, 'diff.txt'))
            for line in lines:
                a, b = line.split(',')
                # to zero-based index
                a = int(a) - 1
                b = int(b) - 1
                diff_list.append([set_a[a], set_b[b]])
        return same_list, diff_list
        
        
    @staticmethod
    def _parse_img_list(path):
        fp_list = []
        for line in readlines(path):
            segs = line.split()
            p = segs[1]
            p = p.replace('../Data/Images/','')
            fp_list.append(p)
        return fp_list
        
        
    def get_pairs(self, proto='FP'):
        pos, neg = self.fp_pairs if proto == 'FP' else self.ff_pairs
        pos = [[os.path.join(self.aligned_dir, p[0]), os.path.join(self.aligned_dir, p[1])] for p in pos]
        neg = [[os.path.join(self.aligned_dir, p[0]), os.path.join(self.aligned_dir, p[1])] for p in neg]
        return pos, neg
        
        
    def fp_size(self):
        return len(self.fp_list)
        
        
    def get_fp_path(self, index):
        return os.path.join(self.img_dir+'/Data/Images', self.fp_list[index])
        
        
    def get_fp_fiducial(self, index):
        img_path = self.fp_list[index]
        #img_path = img_path.replace('Images', 'Fiducials')
        img_path = img_path.replace('.jpg', '.txt')
        return os.path.join(self.img_dir+'/Data/Fiducials', img_path)
  