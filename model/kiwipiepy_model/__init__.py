'''
This package is just to provide model files for `kiwipiepy`
'''

from kiwipiepy_model._version import __version__

def get_model_path():
    import os
    path = os.path.abspath(__file__)
    dir_path = os.path.dirname(path)
    return dir_path
