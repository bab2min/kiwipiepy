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

_tag_set = {'NNG', 'NNP', 'NNB', 'NR', 'NP', 'VV', 'VA', 'VX', 'VCP', 'VCN', 'MM', 'MAG', 'MAJ', 'IC', 'JKS',
             'JKC', 'JKG', 'JKO', 'JKB', 'JKV', 'JKQ', 'JX', 'JC', 'EP', 'EF', 'EC', 'ETN', 'ETM', 'XPN', 'XSN',
             'XSV', 'XSA', 'XR', 'SF', 'SP', 'SS', 'SE', 'SO', 'SW', 'SL', 'SH', 'SN', 'UN', 'W_URL', 'W_EMAIL',
             'W_HASHTAG', 'W_MENTION'}

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

    def _load_stopwords(self, filename):
        stopwords = set()
        stoptags = set()
        for stopword in open(filename, 'r', encoding='utf-8'):
            stopword = stopword.strip()
            try:
                form, tag = stopword.split('/')
            except:
                raise ValueError(f"Line in format 'form/tag' expected, but {repr(stopword)} found.")
            if form: stopwords.add((form, tag))
            else: stoptags.add(tag)
        return stopwords, stoptags

    def _save_stopwords(self, filename, stopwords, stoptags):
        with open(filename, 'w', encoding='utf-8') as fout:
            for tag in stoptags:
                print(f"/{tag}", file=fout)
            for form, tag in stopwords:
                print(f"{form}/{tag}", file=fout)

    def __init__(self, filename=None):
        if filename is None:
            path = os.path.abspath(__file__)
            dir_path = os.path.dirname(path)
            filename = dir_path + '/corpus/stopwords.txt'
        self.stopwords, self.stoptags = self._load_stopwords(filename)

    def save(self, filename):
        self._save_stopwords(filename, self.stopwords, self.stoptags)

    def __contains__(self, word):
        if isinstance(word, str):
            warnings.warn("`word` should be in a tuple of `(form, tag)`.", RuntimeWarning)
        
        try:
            form, tag = word
        except:
            raise ValueError("`word` should be in a tuple of `(form, tag)`.")

        return word in self.stopwords or tag in self.stoptags

    def _tag_exists(self, tag):
        if tag in _tag_set:
            return True
        raise ValueError(f'{repr(tag)} is an invalid tag.')

    def _token_exists(self, token):
        if token in self.stopwords:
            return True
        raise ValueError(f"{repr(token)} doesn't exist in stopwords")

    def _is_not_stopword(self, token):
        return (token.form, token.tag) not in self.stopwords and token.tag not in self.stoptags

    def add(self, tokens):
        '''불용어 사전에 사용자 정의 불용어를 추가합니다.

Parameters
----------
tokens: Union[str, Tuple[str, str]]
    추가할 불용어입니다.
    이 인자는 Iterable로 줄 수 있습니다.
    str로 줄 경우, 품사 태그는 'NNP'로 처리합니다.
    Tuple로 줄 경우, `(단어 형태, 품사 태그)`로 처리합니다.
        '''

        if type(tokens) is str:
            self.stopwords.add((tokens, 'NNP'))
        elif type(tokens) is tuple and self._tag_exists(tokens[1]):
            self.stopwords.add(tokens)
        else:
            for token in tokens:
                if type(token) is str:
                    token = (token, 'NNP')
                    self.stopwords.add(token)
                    continue
                if self._tag_exists(token[1]):
                    self.stopwords.add(token)

    def remove(self, tokens):
        '''불용어 사전에서 입력 불용어를 제거합니다.

Parameters
----------
tokens: Union[str, Tuple[str, str]]
    제거할 불용어입니다.
    이 인자는 Iterable로 줄 수 있습니다.
    str로 줄 경우, 품사 태그는 'NNP'로 처리합니다.
    Tuple로 줄 경우, `(단어 형태, 품사 태그)`로 처리합니다.
        '''

        if type(tokens) is str and self._token_exists((tokens, 'NNP')):
            self.stopwords.remove((tokens, 'NNP'))
        elif type(tokens) is tuple and self._token_exists(tokens):
            self.stopwords.remove(tokens)
        else:
            for token in tokens:
                if type(token) is str:
                    token = (token, 'NNP')
                if self._token_exists(token):
                    self.stopwords.remove(token)

    def filter(self, tokens):
        '''불용어를 필터링합니다.

Parameters
----------
tokens: Iterable[kiwipiepy.Token]
    필터링할 `kiwipiepy.Token`의 리스트, 혹은 Iterable입니다.

Returns
-------
filtered_tokens: List[kiwipiepy.Token]
    필터링 결과를 반환합니다. 리스트의 각 항목은 `kiwipiepy.Token`입니다.
        '''
        
        return list(filter(self._is_not_stopword, tokens))
