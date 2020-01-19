# coding=utf-8
import os
import cv2
import copy
from deeploader.dataset.dataset_base import ArrayDataset

def load_landmarks(path):
    pts = []
    with open(path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip()
            if not line:
                continue
            seg = line.split()
            x = int(seg[0])
            y = int(seg[1])
            pts.append([x, y])
    #return np.array(pts)
    return pts

    
class CelebADataset(ArrayDataset): 
    def __init__(self, rootdir, imgdir = 'img_align_celeba', name='CelebA'):
        ArrayDataset.__init__(self, name)
        self.rootdir = rootdir
        self.attr_path = os.path.join(rootdir, 'list_attr_celeba.txt')
        self.landmarks_path = os.path.join(rootdir, 'list_landmarks_align_celeba.txt')
        self.bbox_path = os.path.join(rootdir, 'list_bbox_celeba.txt')
        self.img_dir = os.path.join(rootdir, imgdir)
        # image list
        self.img_list = os.listdir(self.img_dir)
        # {'path', 'landmark':[], 'attr':[] }
        # parse attr
        with open(self.landmarks_path, 'r') as f:
            total = f.readline()
            total = int(total)
            item_list = [{} for i in range(total)]
            attr_names = f.readline().strip()
            self.attr_names = attr_names.split()
            idx = 0
            for line in f.readlines():
                fields = line.strip().split()
                fname = fields[0]
                attr = [int(x) for x in fields[1:]]
                item = {'path': fname, 'attr': attr}
                item_list[idx] = item
                idx += 1
        # parse landmarks
        with open(self.landmarks_path, 'r') as f:
            f.readline()
            f.readline()
            idx = 0
            for line in f.readlines():
                fields = line.strip().split()
                landmarks = [int(x) for x in fields[1:]]
                item_list[idx]['landmark'] = landmarks
                idx += 1
        # parse bboxs
        with open(self.bbox_path, 'r') as f:
            f.readline()
            f.readline()
            idx = 0
            for line in f.readlines():
                fields = line.strip().split()
                bbox = [int(x) for x in fields[1:]]
                item_list[idx]['bbox'] = bbox
                idx += 1
        # check if got landmark directory
        ldir = os.path.join(rootdir, 'landmark')
        if os.path.exists(ldir):
            files = os.listdir(ldir)
            if len(files) > len(item_list) * 0.8:
                ok_idx = []
                for i in range(total):
                    item = item_list[i]
                    lpath = os.path.join(ldir, item['path']) + '.txt'
                    if not os.path.exists(lpath):
                        continue
                    pts = load_landmarks(lpath)
                    item['kp68'] = pts
                    ok_idx.append(i)
                # filter
                outputs = []
                for idx in ok_idx:
                    outputs.append(item_list[idx])
                item_list = outputs
        self.list = item_list
        
        
    def size(self):
        return len(self.list)
  
  
    def getData(self, index):
        item = copy.deepcopy(self.list[index])
        rel_path = item['path']
        path = os.path.join(self.img_dir, rel_path)
        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        item['img'] = img
        
        return item
  