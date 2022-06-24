"""
.. include:: ./documentation.md
"""
from _kiwipiepy import Token
from kiwipiepy._version import __version__
from kiwipiepy._wrap import Kiwi, Sentence, TypoTransformer, TypoDefinition, basic_typos
import kiwipiepy.utils as utils
from kiwipiepy.const import Match, Option

Kiwi.__module__ = 'kiwipiepy'
TypoTransformer.__module__ = 'kiwipiepy'
TypoDefinition.__module__ = 'kiwipiepy'
Sentence.__module__ = 'kiwipiepy'
