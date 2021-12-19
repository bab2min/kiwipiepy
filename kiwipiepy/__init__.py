"""
.. include:: ./documentation.rst
"""
from _kiwipiepy import Token
from kiwipiepy._version import __version__
from kiwipiepy._wrap import Option, Match, Kiwi, Sentence
import kiwipiepy.utils as utils

Option.__module__ = 'kiwipiepy'
Match.__module__ = 'kiwipiepy'
Kiwi.__module__ = 'kiwipiepy'
Sentence.__module__ = 'kiwipiepy'
