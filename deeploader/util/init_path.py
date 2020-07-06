# -*- coding:utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# init path
import os
import sys


def file_dir(path):
    my = os.path.dirname(os.path.abspath(path))
    return my


def require_(base, *args):
    if os.path.isfile(base):
        base = file_dir(base)
    path = os.path.join(base, *args)
    path = os.path.abspath(path)
    if path not in sys.path:
        sys.path.insert(0, path)


def require(*args):
    p = os.path.abspath(sys._getframe(1).f_code.co_filename)
    p = os.path.dirname(p)
    require_(p, *args)


def mypath():
    p = os.path.abspath(sys._getframe(1).f_code.co_filename)
    return p


def mydir():
    p = os.path.abspath(sys._getframe(1).f_code.co_filename)
    p = os.path.dirname(p)
    return p
