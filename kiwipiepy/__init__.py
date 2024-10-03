"""
.. include:: ./documentation.md
"""
from kiwipiepy._c_api import Token
from kiwipiepy._version import __version__
from kiwipiepy._wrap import Kiwi, Sentence, TypoTransformer, TypoDefinition, HSDataset, MorphemeSet, PretokenizedToken, extract_substrings, NgramExtractor
import kiwipiepy.sw_tokenizer as sw_tokenizer
import kiwipiepy.utils as utils
from kiwipiepy.const import Match
from kiwipiepy.default_typo_transformer import basic_typos, continual_typos, basic_typos_with_continual, lengthening_typos, basic_typos_with_continual_and_lengthening

Kiwi.__module__ = 'kiwipiepy'
TypoTransformer.__module__ = 'kiwipiepy'
TypoDefinition.__module__ = 'kiwipiepy'
Sentence.__module__ = 'kiwipiepy'
PretokenizedToken.__module__ = 'kiwipiepy'
HSDataset.__module__ = 'kiwipiepy'
MorphemeSet.__module__ = 'kiwipiepy'
