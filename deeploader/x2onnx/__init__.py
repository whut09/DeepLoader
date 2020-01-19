import os
import sys

proj_dir = os.path.dirname(os.path.abspath(__file__))

# module path
_path_installed = False


def init_path():
    global _path_installed
    if _path_installed:
        return
    sys.path.append(proj_dir)
    # print(sys.path)
    _path_installed = True


init_path()
