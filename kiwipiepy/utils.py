'''
utils 모듈은 kiwipiepy를 사용하는 데에 있어서 다양한 편의 기능을 제공하기 위한 유틸리티성 클래스 및 함수를 제공합니다.
현재는 Stopwords 클래스만 포함되어 있으며, 이 클래스는 불용어를 관리하고 Kiwi의 형태소 분석 결과 중 불용어를 쉽게 필터링하는 기능을 제공합니다.
이 기능은 [HyeJuSeon](https://github.com/HyeJuSeon/)님의 기여로 추가되었습니다.

```python
from kiwipiepy import Kiwi
from kiwipiepy.utils import Stopwords

kiwi = Kiwi()
stopwords = Stopwords()

print(kiwi.tokenize("나는 학교에 방문한다"))
#[Token(form='나', tag='NP', start=0, len=1),
# Token(form='는', tag='JX', start=1, len=1),
# Token(form='학교', tag='NNG', start=3, len=2),
# Token(form='에', tag='JKB', start=5, len=1),
# Token(form='방문', tag='NNG', start=7, len=2),
# Token(form='하', tag='XSV', start=9, len=1),
# Token(form='ᆫ다', tag='EC', start=10, len=1)]

print(stopwords.filter(kiwi.tokenize("나는 학교에 방문한다")))
#[Token(form='학교', tag='NNG', start=3, len=2),
# Token(form='방문', tag='NNG', start=7, len=2),
# Token(form='ᆫ다', tag='EC', start=10, len=1)]
```
'''

import os
import warnings

_default_stopwords_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'corpus', 'stopwords.txt')

_default_tag = 'NNP'

_valid_tags = frozenset((
    'NNG', 'NNP', 'NNB', 'NR', 'NP',
    'VV', 'VA', 'VX', 'VCP', 'VCN',
    'MM', 'MAG', 'MAJ', 'IC',
    'JKS', 'JKC', 'JKG', 'JKO', 'JKB', 'JKV', 'JKQ', 'JX', 'JC',
    'EP', 'EF', 'EC', 'ETN', 'ETM',
    'XPN', 'XSN', 'XSV', 'XSA', 'XSM', 'XR',
    'SF', 'SP', 'SS', 'SSO', 'SSC', 'SE', 'SO', 'SW', 'SB',
    'SL', 'SH', 'SN', 'UN',
    'W_URL', 'W_EMAIL', 'W_HASHTAG', 'W_MENTION', 'W_SERIAL', 'W_EMOJI',
    'Z_CODA', 'Z_SIOT',
    'USER0', 'USER1', 'USER2', 'USER3', 'USER4',
))

def _check_tag(tag):
    # 불규칙 활용 태그(VV-I 등)는 규칙 활용 태그와 동일하게 취급한다.
    if not isinstance(tag, str) or tag.split('-')[0] not in _valid_tags:
        raise ValueError(f'{repr(tag)} is an invalid tag.')

def _to_pair(token):
    if isinstance(token, str):
        return token, _default_tag

    try:
        form, tag = token
    except (TypeError, ValueError):
        raise ValueError(f"`str` or a tuple of `(form, tag)` expected, but {repr(token)} found.")
    _check_tag(tag)
    return form, tag

def _to_pairs(tokens):
    if isinstance(tokens, str):
        yield _to_pair(tokens)
    elif isinstance(tokens, tuple) and len(tokens) == 2 and all(isinstance(i, str) for i in tokens):
        yield _to_pair(tokens)
    else:
        for token in tokens:
            yield _to_pair(token)

class Stopwords:
    '''
    .. versionadded:: 0.10.2

불용어를 관리하는 유틸리티 클래스입니다.

Parameters
----------
filename: str
    읽어들일 불용어 파일의 경로. 생략하거나 None을 줄 경우 kiwipiepy에 내장된 기본 불용어 사전을 불러옵니다.
    기본 불용어 사전은 AIHub & 모두의 말뭉치 코퍼스를 이용해서 tf 기준 상위 100개를 추출하여 구축되었습니다.
    '''

    def __init__(self, filename=None):
        self.stopwords = set()
        self.stoptags = set()

        with open(filename if filename is not None else _default_stopwords_path, encoding='utf-8') as fin:
            for line in fin:
                entry = line.strip()
                if not entry: continue
                form, sep, tag = entry.rpartition('/')
                if not sep:
                    raise ValueError(f"Line in format 'form/tag' expected, but {repr(entry)} found.")
                # 형태 없이 태그만 적힌 줄은 해당 품사 전체를 불용 처리한다.
                if form: self.stopwords.add((form, tag))
                else: self.stoptags.add(tag)

    def save(self, filename):
        '''현재 불용어 사전을 파일로 저장합니다.

Parameters
----------
filename: str
    저장할 파일의 경로입니다. 저장된 파일은 `Stopwords`의 생성자로 다시 읽어들일 수 있습니다.
        '''

        with open(filename, 'w', encoding='utf-8') as fout:
            for tag in sorted(self.stoptags):
                fout.write(f'/{tag}\n')
            for form, tag in sorted(self.stopwords):
                fout.write(f'{form}/{tag}\n')

    def __contains__(self, word):
        if isinstance(word, str):
            warnings.warn("`word` should be in a tuple of `(form, tag)`.", RuntimeWarning)

        try:
            form, tag = word
        except (TypeError, ValueError):
            raise ValueError("`word` should be in a tuple of `(form, tag)`.")

        return (form, tag) in self.stopwords or tag in self.stoptags

    def add(self, tokens):
        '''불용어 사전에 새로운 항목을 등록합니다.

Parameters
----------
tokens: Union[str, Tuple[str, str]]
    추가할 불용어입니다. 여러 개를 추가하려면 Iterable로 줄 수 있습니다.
    `str`로 준 경우 품사 태그가 'NNP'인 것으로 간주하며,
    `tuple`로 준 경우 `(형태, 품사 태그)`로 간주합니다.
        '''

        self.stopwords.update(_to_pairs(tokens))

    def remove(self, tokens):
        '''불용어 사전에 등록된 항목을 지웁니다.

Parameters
----------
tokens: Union[str, Tuple[str, str]]
    제거할 불용어입니다. 여러 개를 제거하려면 Iterable로 줄 수 있습니다.
    `str`로 준 경우 품사 태그가 'NNP'인 것으로 간주하며,
    `tuple`로 준 경우 `(형태, 품사 태그)`로 간주합니다.
    사전에 없는 불용어를 제거하려고 하면 `ValueError`가 발생합니다.
        '''

        for pair in _to_pairs(tokens):
            if pair not in self.stopwords:
                raise ValueError(f"{repr(pair)} doesn't exist in stopwords")
            self.stopwords.discard(pair)

    def filter(self, tokens):
        '''형태소 분석 결과에서 불용어를 제거한 결과를 반환합니다.

Parameters
----------
tokens: Iterable[kiwipiepy.Token]
    걸러낼 `kiwipiepy.Token`의 리스트 혹은 Iterable입니다.

Returns
-------
filtered_tokens: List[kiwipiepy.Token]
    불용어를 제외한 나머지 토큰의 리스트입니다. 각 항목은 `kiwipiepy.Token`입니다.
        '''

        return [token for token in tokens
                if (token.form, token.tag) not in self.stopwords and token.tag not in self.stoptags]
