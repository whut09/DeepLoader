# -*- coding:utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
import os.path
import time


def translate_path(src_prefix, dst_prefix, path):
    sp = ''
    dp = ''
    if src_prefix:
        sp = src_prefix.replace('\\', '/')
    if dst_prefix:
        dp = dst_prefix.replace('\\', '/')
    p = path.replace('\\', '/')
    # src -> dst
    if p.startswith(sp):
        rel_p = p[len(sp):len(p)]
        # print(rel_p)
        return dp + rel_p
    # dst -> src
    if p.startswith(dp):
        rel_p = p[len(dp):len(p)]
        return sp + rel_p
    return dp + p


def makedirs(path):
    dir, fname = os.path.split(path)
    if not os.path.exists(dir):
        try:
            os.makedirs(dir)
        except:
            pass


def makedir(path):
    try:
        os.makedirs(path)
    except:
        pass


def file_walker(dir, visitor):
    '''
        Recursive walk through a dir
    '''
    filenames = os.listdir(dir)
    for filename in filenames:
        fullpath = os.path.join(dir, filename)
        if os.path.isdir(fullpath):
            file_walker(fullpath, visitor)
        elif os.path.isfile(fullpath):
            visitor.process(fullpath)


def read_lines(path):
    list = []
    if not os.path.exists(path):
        return list

    with open(path, 'r') as f:
        for line in f.readlines():
            list.append(line.strip())
    print('read:%d lines from %s' % (len(list), path))
    return list


def read_json(path):
    list = ""
    if not os.path.exists(path):
        return list

    with open(path, 'r') as f:
        for line in f.readlines():
            list = list + line.strip()
    # print('read:%d chars from %s' % (len(list), path))
    return list


def write_lines(path, lines):
    with open(path, 'w') as f:
        for line in lines:
            f.write('%s\n' % line)
    print('write:%d lines to %s' % (len(lines), path))
    return lines


def list_walker(list_path, visitor):
    list = read_lines(list_path)
    for i in range(len(list)):
        path = list[i]
        visitor.process(path)


def get_file_md5(filepath):
    # 获取文件的md5
    if not os.path.isfile(filepath):
        return
    myhash = hashlib.md5()
    f = open(filepath, "rb")
    while True:
        b = f.read(8096)
        if not b:
            break
        myhash.update(b)
    f.close()
    # print(myhash.hexdigest())
    return myhash.hexdigest()


def get_file_size(filepath):
    '''
    Get file size in bytes
    :param filepath: file path
    :return: file size in bytes
    '''
    try:
        file_size = os.path.getsize(filepath)
        return file_size
    except Exception as err:
        print(err)
        return 0


IMAGE_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'gif', 'bmp', 'tif'])


def allowed_file(filename, exts=None):
    if exts == None:
        exts = IMAGE_EXTENSIONS

    return '.' in filename and filename.rsplit('.', 1)[1].lower() in exts


def is_image_file(path):
    return allowed_file(path)

    
def clean_dir(rootdir):
    if not os.path.exists(rootdir):
        return
    import shutil
    filelist = os.listdir(rootdir)
    for f in filelist:
        filepath = os.path.join(rootdir, f)
        if os.path.isfile(filepath):
            os.remove(filepath)
        elif os.path.isdir(filepath):
            shutil.rmtree(filepath, True)    