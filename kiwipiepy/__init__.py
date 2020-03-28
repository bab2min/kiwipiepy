from _kiwipiepy import *
from enum import IntEnum

class Option(IntEnum):
    LOAD_DEFAULT_DICTIONARY = 1
    INTEGRATE_ALLOMORPH = 2

class Match(IntEnum):
    URL = 1
    EMAIL = 2
    HASHTAG = 4
    ALL = 7

del IntEnum