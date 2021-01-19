from setuptools import setup, Extension 
from codecs import open
import os, os.path, platform
from setuptools.command.install import install

here = os.path.abspath(os.path.dirname(__file__))
exec(open('kiwipiepy_model/_version.py').read())

long_description = '''Model for kiwipiepy
-------------------
kiwipiepy is a python version package of Kiwi(Korean Intelligent Word Identifier) which is a morphological analyzer for Korean.

https://github.com/bab2min/kiwipiepy '''

setup(
    name='kiwipiepy_model',
    version=__version__,

    description='Model for kiwipiepy',
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

    packages=['kiwipiepy_model'],
    include_package_data=True,
    zip_safe=False,
)
