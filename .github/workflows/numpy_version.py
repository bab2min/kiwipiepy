import sys

def get_old_numpy_version(use_v1=False):
    py_version = sys.version_info
    if not use_v1:
        if py_version >= (3, 10): return '2.1.*'
        if py_version >= (3, 9): return '2.0.*'
    if py_version >= (3, 13): return '2.1.*'
    if py_version >= (3, 12): return '1.26.0'
    if py_version >= (3, 11): return '1.24.0'
    if py_version >= (3, 10): return '1.22.0'
    if py_version >= (3, 9): return '1.20.0'
    if py_version >= (3, 8): return '1.18.0'
    if py_version >= (3, 7): return '1.15.0'
    if py_version >= (3, 6): return '1.12.0'
    if py_version >= (3, 5): return '1.11.0'
    return '1.10.0'

if __name__ == '__main__':
    use_v1 = len(sys.argv) > 1 and sys.argv[1] == 'v1'
    print(get_old_numpy_version(use_v1))
