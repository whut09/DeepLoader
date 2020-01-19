# coding:utf-8
from __future__ import print_function
import numpy as np
import os
import sys
import time
import torch
from deeploader.util.fileutil import makedirs


def resume_checkpoint(ckpt_path, net):
    if os.path.exists(ckpt_path):
        print('Resume from:%s' % ckpt_path)
        ckpt = torch.load(ckpt_path)
        if 'net_state_dict' in ckpt:
            state_dict = ckpt['net_state_dict']
        else:
            state_dict = ckpt
        try:
            net.load_state_dict(state_dict)
        except:
            mod_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
            net.load_state_dict(mod_dict, strict=False)
            # net.load_state_dict(mod_dict)
        return ckpt
    else:
        print('Checkpoint not found:%s' % ckpt_path)
        return None


def save_checkpoint(ckpt_prefix, net, step, epoch):
    msg = 'Saving checkpoint @ step:{} epoch:{}'.format(step, epoch)
    print(msg)
    if isinstance(net, torch.nn.DataParallel):
        net_state_dict = net.module.state_dict()
    else:
        net_state_dict = net.state_dict()
    _fname = 'step-%05d_epoch-%03d.ckpt' % (step, epoch)
    save_path = os.path.join(ckpt_prefix, _fname)
    makedirs(save_path)
    torch.save({
        'step': step,
        'epoch': epoch,
        'net_state_dict': net_state_dict},
        save_path)


def compute_acc(logits, labels):
    batch_size = logits.size(0)
    _, preds = torch.max(logits, 1)
    running_corrects = torch.sum(preds == labels.data)
    _acc = running_corrects.double() / batch_size
    return _acc


def image_to_tensor(img):
    img = np.float32(img)
    img = torch.from_numpy(img.transpose((0, 3, 1, 2)))
    return img


def label_to_tensor(label):
    label = label.astype(np.int64)
    label = torch.from_numpy(label)
    return label


def to_one_hot(idx, num_classes):
    return torch.zeros(len(idx), num_classes).scatter_(1, idx.unsqueeze(1), 1.)
