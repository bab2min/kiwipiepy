import os, os.path
from glob import glob
import platform
import sys
import shutil
import subprocess
import re

from distutils import log
from setuptools.command.install import install
from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext
from distutils.version import LooseVersion

import numpy as np

if os.environ.get('KIWI_CPU_ARCH'):
    from sysconfig import get_platform
    fd = get_platform().split('-')
    if fd[0] == 'macosx':
        os.environ['_PYTHON_HOST_PLATFORM'] = '-'.join(fd[:-1] + [os.environ['KIWI_CPU_ARCH']])

def get_extra_cmake_options():
    """read --clean, --no, --set, --compiler-flags, and -G options from the command line and add them as cmake switches.
    """
    _cmake_extra_options = ["-DKIWI_BUILD_TEST=0", "-DCMAKE_POSITION_INDEPENDENT_CODE=1", "-DKIWI_USE_MIMALLOC=" + ("1" if os.environ.get('USE_MIMALLOC') else "0")]
    if os.environ.get('KIWI_CPU_ARCH'):
        _cmake_extra_options.append("-DKIWI_CPU_ARCH=" + os.environ['KIWI_CPU_ARCH'])
    if os.environ.get('MACOSX_DEPLOYMENT_TARGET'):
        _cmake_extra_options.append("-DCMAKE_OSX_DEPLOYMENT_TARGET=" + os.environ['MACOSX_DEPLOYMENT_TARGET'])
    _clean_build_folder = False
    print(_cmake_extra_options)

    opt_key = None

    argv = [arg for arg in sys.argv]  # take a copy
    # parse command line options and consume those we care about
    for arg in argv:
        if opt_key == 'compiler-flags':
            _cmake_extra_options.append('-DCMAKE_CXX_FLAGS={arg}'.format(arg=arg.strip()))
        elif opt_key == 'G':
            _cmake_extra_options += ['-G', arg.strip()]
        elif opt_key == 'no':
            _cmake_extra_options.append('-D{arg}=no'.format(arg=arg.strip()))
        elif opt_key == 'set':
            _cmake_extra_options.append('-D{arg}'.format(arg=arg.strip()))

        if opt_key:
            sys.argv.remove(arg)
            opt_key = None
            continue

        if arg == '--clean':
            _clean_build_folder = True
            sys.argv.remove(arg)
            continue

        if arg == '--yes':
            print("The --yes options to kiwipiepy's setup.py don't do anything since all these options ")
            print("are on by default.  So --yes has been removed.  Do not give it to setup.py.")
            sys.exit(1)
        if arg in ['--no', '--set', '--compiler-flags']:
            opt_key = arg[2:].lower()
            sys.argv.remove(arg)
            continue
        if arg in ['-G']:
            opt_key = arg[1:]
            sys.argv.remove(arg)
            continue

    return _cmake_extra_options, _clean_build_folder

def num_available_cpu_cores(ram_per_build_process_in_gb):
    import multiprocessing
    try:
        mem_bytes = os.sysconf('SC_PAGE_SIZE') * os.sysconf('SC_PHYS_PAGES')  
        mem_gib = mem_bytes/(1024.**3)
        num_cores = multiprocessing.cpu_count() 
        mem_cores = int(mem_gib/float(ram_per_build_process_in_gb)+0.5)
        return max(min(num_cores, mem_cores), 1)
    except ValueError:
        return 2 # just assume 2 if we can't get the os to tell us the right answer.

cmake_extra_options, clean_build_folder = get_extra_cmake_options()


class CMakeExtension(Extension):
    def __init__(self, name, sourcedir='', *args, **kwargs):
        Extension.__init__(self, name, sources=[], *args, **kwargs)
        self.sourcedir = os.path.abspath(sourcedir)

class CMakeBuild(build_ext):

    def get_cmake_version(self):
        try:
            out = subprocess.check_output(['cmake', '--version'])
        except:
            sys.stderr.write("\nERROR: CMake must be installed to build kiwipiepy\n\n") 
            sys.exit(1)
        return re.search(r'version\s*([\d.]+)', out.decode()).group(1)

    def run(self):
        cmake_version = self.get_cmake_version()
        if platform.system() == "Windows":
            if LooseVersion(cmake_version) < '3.1.0':
                sys.stderr.write("\nERROR: CMake >= 3.1.0 is required on Windows\n\n")
                sys.exit(1)

        for ext in self.extensions:
            self.build_extension(ext)

    def build_extension(self, ext):
        extdir = os.path.abspath(os.path.dirname(self.get_ext_fullpath(ext.name)))
        libs = self.get_libraries(ext)

        cmake_args = [
            '-DINCLUDE_DIRS={}'.format(';'.join(self.include_dirs + [np.get_include()])),
            '-DLIBRARY_DIRS={}'.format(';'.join(self.library_dirs)),
            '-DLIBRARIES={}'.format(';'.join(libs)),
            '-DPYTHON_EXECUTABLE=' + sys.executable,
        ]
        print(cmake_args)
        cmake_args += cmake_extra_options

        cfg = 'Debug' if self.debug else 'Release'
        build_args = ['--config', cfg]

        if platform.system() == "Windows":
            if sys.maxsize > 2**32:
                cmake_args += ['-A', 'x64']
            else:
                cmake_args += ['-A', 'Win32']
            build_args += ['--', '/m'] 
        else:
            cmake_args += ['-DCMAKE_BUILD_TYPE=' + cfg]
            build_args += ['--', '-j'+str(num_available_cpu_cores(2))]

        build_folder = os.path.abspath(self.build_temp)

        if not os.path.exists(build_folder):
            os.makedirs(build_folder)

        cmake_setup = ['cmake', ext.sourcedir] + cmake_args
        cmake_build = ['cmake', '--build', '.'] + build_args

        print("Building extension for Python {}".format(sys.version.split('\n',1)[0]))
        print("Invoking CMake setup: '{}'".format(' '.join(cmake_setup)))
        sys.stdout.flush()
        subprocess.check_call(cmake_setup, cwd=build_folder)
        print("Invoking CMake build: '{}'".format(' '.join(cmake_build)))
        sys.stdout.flush()
        subprocess.check_call(cmake_build, cwd=build_folder)
        
        fullname = self.get_ext_fullname(ext.name)
        filename = self.get_ext_filename(fullname)

        if platform.system() == "Windows":
            shutil.move(os.sep.join([build_folder, cfg, '_kiwipiepy.dll']), os.sep.join([extdir, filename]))
        else:
            shutil.move(glob(os.sep.join([build_folder, 'lib_kiwipiepy.*']))[0], os.sep.join([extdir, filename]))
        #print()


here = os.path.abspath(os.path.dirname(__file__))
exec(open('kiwipiepy/_version.py').read())

long_description = '''kiwipiepy
----------
kiwipiepy is a python version package of Kiwi(Korean Intelligent Word Identifier) which is a morphological analyzer for Korean.

https://github.com/bab2min/kiwipiepy '''


libraries = []

if platform.system() == 'Windows': 
    if os.environ.get('USE_MIMALLOC'): libraries.append('advapi32.lib')
else: 
    pass

setup(
    name='kiwipiepy',
    version=__version__,

    description='Kiwi, the Korean Tokenizer for Python',
    long_description=long_description,

    url='https://github.com/bab2min/kiwipiepy',

    author='bab2min',
    author_email='bab2min@gmail.com',

    license='LGPL v3 License',

    classifiers=[
        'Development Status :: 3 - Alpha',

        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Topic :: Software Development :: Libraries",
        "Topic :: Text Processing :: Linguistic",

        "License :: OSI Approved :: GNU Lesser General Public License v3 (LGPLv3)",

        'Programming Language :: Python :: 3',
        'Programming Language :: C++'
    ],

    keywords='Korean morphological analysis',
    install_requires=[
        'dataclasses; python_version < "3.7"',
        'kiwipiepy_model>=0.19,<0.20',
        'numpy<2; python_version < "3.9"',
        'numpy; python_version >= "3.9"',
        'tqdm',
    ],
    packages=['kiwipiepy'],
    include_package_data=True,
    ext_modules=[CMakeExtension('_kiwipiepy',
        libraries=libraries,
    )],
    cmdclass=dict(build_ext=CMakeBuild),
)
