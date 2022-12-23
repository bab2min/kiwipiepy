import re
from typing import Callable, List, Optional, Tuple, Union, Iterable
from collections import namedtuple
from dataclasses import dataclass
import warnings

from _kiwipiepy import _Kiwi, _TypoTransformer, _HSDataset
from kiwipiepy._version import __version__
from kiwipiepy.utils import Stopwords
from kiwipiepy.const import Match, Option

Sentence = namedtuple('Sentence', ['text', 'start', 'end', 'tokens', 'subs'])
Sentence.__doc__ = '문장 분할 결과를 담기 위한 `namedtuple`입니다.'
Sentence.text.__doc__ = '분할된 문장의 텍스트'
Sentence.start.__doc__ = '전체 텍스트 내에서 분할된 문장이 시작하는 위치 (문자 단위)'
Sentence.end.__doc__ = '전체 텍스트 내에서 분할된 문장이 끝나는 위치 (문자 단위)'
Sentence.tokens.__doc__ = '분할된 문장의 형태소 분석 결과'
Sentence.subs.__doc__ = '''.. versionadded:: 0.14.0

현 문장 내에 포함된 안긴 문장의 목록
'''

@dataclass
class TypoDefinition:
    '''.. versionadded:: 0.13.0

오타 생성 규칙을 정의하는 dataclass

Parameters
----------
orig: List[str]
    원본 문자열
error: List[str]
    교체될 오타의 문자열
cost: float
    교체 비용
condition: str
    오타 교체가 가능한 환경 조건. `None`, `'any'`(아무 글자 뒤), `'vowel'`(모음 뒤), `'applosive'`(불파음 뒤) 중 하나. 생략 시 기본값은 `None`입니다.

Notes
-----
`orig`나 `error`는 완전한 음절 혹은 모음이나 자음을 포함할 수 있습니다. 자음의 경우 종성은 '\\'로 escape해주어야 합니다.

```python
TypoDefinition(['개'], ['게'], 1.0) # '개'를 '게'로 교체
TypoDefinition(['ㅐ'], ['ㅔ'], 1.0) # 모든 'ㅐ'를 'ㅒ'로 교체
TypoDefinition(['ㄲ'], ['ㄱ'], 1.0) # 모든 초성 'ㄲ'을 초성 'ㄱ'으로 교체
TypoDefinition(['ㄳ'], ['ㄱ'], 1.0) # 'ㄳ'에 해당하는 초성은 없으므로 ValueError 발생
TypoDefinition(['\ㄳ'], ['\ㄱ'], 1.0) # 모든 종성 'ㄳ'을 종성 'ㄱ'으로 교체
```
    '''
    orig: List[str]
    error: List[str]
    cost: float
    condition: Optional[str] = None

    def __post_init__(self):
        if self.condition not in (None, 'any', 'vowel', 'applosive'):
            raise ValueError("`condition` should be one of (None, 'any', 'vowel', 'applosive'), but {}".format(self.condition))

_c_to_onset = dict(zip(
    'ㄱㄲㄴㄷㄸㄹㅁㅂㅃㅅㅆㅇㅈㅉㅊㅋㅌㅍㅎ', 
    range(19),
))

_c_to_coda = dict(zip(
    'ㄱㄲㄳㄴㄵㄶㄷㄹㄺㄻㄼㄽㄾㄿㅀㅁㅂㅄㅅㅆㅇㅈㅊㅋㅌㅍㅎ', 
    range(28)
))

def _convert_consonant(s):
    ret = []
    prev_escape = False
    for c in s:
        if prev_escape:
            if c == '\\': ret.append('\\')
            elif c in _c_to_coda: ret.append(chr(0x11A8 + _c_to_coda[c]))
            elif c in _c_to_onset:
                raise ValueError("Wrong consonant '\\{}'".format(c))
            else:
                raise ValueError("Wrong escape chr '\\{}'".format(c))
            prev_escape = False
        else:
            if c == '\\': prev_escape = True
            elif c in _c_to_onset: ret.append(chr(0x1100 + _c_to_onset[c]))
            elif c in _c_to_coda:
                raise ValueError("Wrong consonant {}".format(c))
            else:
                ret.append(c)
    return ''.join(ret)

class TypoTransformer(_TypoTransformer):
    '''.. versionadded:: 0.13.0
    
오타 교정 기능에 사용되는 오타 생성기를 정의합니다.

Parameters
----------
defs: List[TypoDefinition]
    오타 생성 규칙을 정의하는 TypoDefinition의 List입니다.

Notes
-----
이 클래스의 인스턴스를 Kiwi 생성시의 typos 인자로 주면 Kiwi의 오타 교정 기능이 활성화됩니다.
```python
>> from kiwipiepy import Kiwi, TypoTransformer, TypoDefinition
>> typos = TypoTransformer([
    TypoDefinition(["ㅐ", "ㅔ"], ["ㅐ", "ㅔ"], 1.), # ㅐ 혹은 ㅖ를 ㅐ 혹은 ㅖ로 교체하여 오타를 생성. 생성 비용은 1
    TypoDefinition(["ㅔ"], ["ㅖ"], 2.), # ㅔ를 ㅖ로 교체하여 오타를 생성. 생성 비용은 2
])
>> typos.generate('과제', 1.) # 생성 비용이 1.0이하인 오타들을 생성
[('과제', 0.0), ('과재', 1.0)]
>> typos.generate('과제', 2.) # 생성 비용이 2.0이하인 오타들을 생성
[('과제', 0.0), ('과재', 1.0), ('과졔', 2.0)]

>> kiwi = Kiwi(typos=typos, typo_cost_threshold=2.) # typos에 정의된 오타들을 교정 후보로 삼는 Kiwi 생성.
>> kiwi.tokenize('과재를 했다') 
[Token(form='과제', tag='NNG', start=0, len=2), 
 Token(form='를', tag='JKO', start=2, len=1), 
 Token(form='하', tag='VV', start=4, len=1), 
 Token(form='었', tag='EP', start=4, len=1), 
 Token(form='다', tag='EF', start=5, len=1)]
```
    '''

    def __init__(self,
        defs: List[TypoDefinition]
    ):
        self._defs = list(defs)
        super().__init__(
            ((list(map(_convert_consonant, d.orig)), list(map(_convert_consonant, d.error)), d.cost, d.condition) for d in self._defs)
        )

    def generate(self, text:str, cost_threshold:float = 2.5) -> List[Tuple[str, float]]:
        '''입력 텍스트로부터 오타를 생성합니다.

Parameters
----------
text: str
    원본 텍스트
cost_threshold: float
    생성 가능한 오타의 최대 비용

Returns
-------
errors: List[Tuple[str, float]]
    생성된 오타와 그 생성 비용의 List
        '''
        return super().generate(text, cost_threshold)

    @property
    def defs(self):
        '''현재 오타 생성기의 정의자 목록'''
        return self._defs
    
    def __repr__(self):
        return "TypoTransformer([{}])".format(",\n  ".join(map(repr, self._defs)))

class HSDataset(_HSDataset):
    pass

class Kiwi(_Kiwi):
    '''Kiwi 클래스는 실제 형태소 분석을 수행하는 kiwipiepy 모듈의 핵심 클래스입니다.

Parameters
----------
num_workers: int
    내부적으로 멀티스레딩에 사용할 스레드 개수. 0으로 설정시 시스템 내 가용한 모든 코어 개수만큼 스레드가 생성됩니다.
    멀티스레딩은 extract 계열 함수에서 단어 후보를 탐색할 때와 perform, analyze 함수에서만 사용됩니다.
model_path: str
    읽어들일 모델 파일의 경로. 모델 파일의 위치를 옮긴 경우 이 값을 지정해주어야 합니다.
options: int
    Kiwi 생성시의 옵션을 설정합니다. 옵션에 대해서는 `kiwipiepy.const.Option`을 확인하십시오.
    .. deprecated:: 0.10.0
        차기 버전에서 제거될 예정입니다. `options` 대신 `integrate_allormoph` 및 `load_default_dict`를 사용해주세요.

integrate_allormoph: bool
    True일 경우 음운론적 이형태를 통합하여 출력합니다. /아/와 /어/나 /았/과 /었/ 같이 앞 모음의 양성/음성에 따라 형태가 바뀌는 어미들을 하나로 통합하여 출력합니다. 기본값은 True입니다.
load_default_dict: bool
    True일 경우 인스턴스 생성시 자동으로 기본 사전을 불러옵니다. 기본 사전은 위키백과와 나무위키에서 추출된 고유 명사 표제어들로 구성되어 있습니다. 기본값은 True입니다.
load_typo_dict: bool
    .. versionadded:: 0.14.0
    
    True일 경우 인스턴스 생성시 자동으로 내장 오타 사전을 불러옵니다. 오타 사전은 자주 틀리는 오타 일부와 인터넷에서 자주 쓰이는 변형된 종결 어미로 구성되어 있습니다. 기본값은 True입니다.
model_type: str
    .. versionadded:: 0.13.0

    형태소 분석에 사용할 언어 모델을 지정합니다. `'knlm'`, `'sbg'` 중 하나를 선택할 수 있습니다. 기본값은 `'knlm'`입니다. 각 모델에 대한 자세한 설명은 <a href='#_4'>여기</a>를 참조하세요.

typos: Union[str, TypoTransformer]
    .. versionadded:: 0.13.0

    교정에 사용할 오타 정보입니다. 기본값은 `None`으로 이 경우 오타 교정을 사용하지 않습니다. `'basic'`으로 입력시 내장된 기본 오타 정보를 이용합니다.
    이에 대한 자세한 내용은 `kiwipiepy.TypoTransformer` 및 <a href='#_5'>여기</a>를 참조하세요.
typo_cost_threshold: float
    .. versionadded:: 0.13.0

    오타 교정시 고려할 최대 오타 비용입니다. 이 비용을 넘어서는 오타에 대해서는 탐색하지 않습니다. 기본값은 2.5입니다.
    '''

    def __init__(self, 
        num_workers:Optional[int] = None,
        model_path:Optional[str] = None,
        options:Optional[int] = None,
        integrate_allomorph:Optional[bool] = None,
        load_default_dict:Optional[bool] = None,
        load_typo_dict:Optional[bool] = None,
        model_type:Optional[str] = 'knlm',
        typos:Optional[Union[str, TypoTransformer]] = None,
        typo_cost_threshold:Optional[float] = 2.5,
    ) -> None:
        if num_workers is None:
            num_workers = 0
        
        if options is None:
            options = 3
        else:
            warnings.warn(
                "Argument `options` will be removed in future version. Use `integrate_allomorph` or `load_default_dict` instead.",
                DeprecationWarning                
            )
        
        if integrate_allomorph is None:
            integrate_allomorph = bool(options & Option.INTEGRATE_ALLOMORPH)
        if load_default_dict is None:
            load_default_dict = bool(options & Option.LOAD_DEFAULT_DICTIONARY)
        if load_typo_dict is None:
            load_typo_dict = True

        if model_type not in ('knlm', 'sbg'):
            raise ValueError("`model_type` should be one of ('knlm', 'sbg'), but {}".format(model_type))
        
        if typos == 'basic': 
            import kiwipiepy
            typos = kiwipiepy.basic_typos

        super().__init__(
            num_workers=num_workers,
            model_path=model_path,
            integrate_allomorph=integrate_allomorph,
            load_default_dict=load_default_dict,
            load_typo_dict=load_typo_dict,
            sbg=(model_type=='sbg'),
            typos=typos,
            typo_cost_threshold=typo_cost_threshold,
        )

        self._ns_integrate_allomorph = integrate_allomorph
        self._ns_cutoff_threshold = 8.
        self._ns_unk_form_score_scale = 3.
        self._ns_unk_form_score_bias = 5.
        self._ns_space_penalty = 7.
        self._ns_max_unk_form_size = 6
        self._ns_space_tolerance = 0
        self._ns_typo_cost_weight = 6.
        self._ns_model_type = model_type

    def add_user_word(self,
        word:str,
        tag:Optional[str] = 'NNP',
        score:Optional[float] = 0.,
        orig_word:Optional[str] = None,
    ) -> bool:
        '''현재 모델에 사용자 정의 형태소를 추가합니다.

Parameters
----------
word: str
    추가할 형태소
tag: str
    추가할 형태소의 품사 태그
score: float
    추가할 형태소의 가중치 점수. 
    해당 형태에 부합하는 형태소 조합이 여러 개가 있는 경우, 이 점수가 높을 단어가 더 우선권을 가집니다.
orig_word : str
    .. versionadded:: 0.11.0

    추가할 형태소의 원본 형태소.
    추가할 형태소가 특정 형태소의 변이형인 경우 이 인자로 원본 형태소를 넘겨줄 수 있습니다. 없는 경우 생략할 수 있습니다.
    `orig_word`가 주어진 경우 현재 사전 내에 `orig_word`/`tag` 조합의 형태소가 반드시 존재해야 하며, 그렇지 않으면 `ValueError` 예외를 발생시킵니다.

Returns
-------
inserted: bool
    사용자 정의 형태소가 정상적으로 삽입된 경우 True, 이미 동일한 형태소가 존재하여 삽입되지 않은 경우 False를 반환합니다.
        '''
        if re.search(r'\s', word): raise ValueError("Whitespace characters are not allowed at `word`")
        return super().add_user_word(word, tag, score, orig_word)
    
    def add_pre_analyzed_word(self,
        form:str,
        analyzed:Iterable[Tuple[str, str]],
        score:Optional[float] = 0.,
    ) -> bool:
        '''.. versionadded:: 0.11.0

현재 모델에 기분석 형태소열을 추가합니다.

Parameters
----------
form: str
    추가할 형태
analyzed: Iterable[Tuple[str, str]]
    `form`의 형태소 분석 결과.
    이 값은 (형태, 품사) 모양의 tuple, 혹은 (형태, 품사, 시작지점, 끝지점) 모양의 tuple로 구성된 Iterable이어야합니다.
    이 값으로 지정되는 형태소는 현재 사전 내에 반드시 존재해야 하며, 그렇지 않으면 `ValueError` 예외를 발생시킵니다.
score: float
    추가할 형태소열의 가중치 점수. 
    해당 형태에 부합하는 형태소 조합이 여러 개가 있는 경우, 이 점수가 높을 단어가 더 우선권을 가집니다.

Returns
-------
inserted: bool
    기분석 형태소열이 정상적으로 삽입된 경우 True, 이미 동일한 대상이 존재하여 삽입되지 않은 경우 False를 반환합니다.

Notes
-----
이 메소드는 불규칙적인 분석 결과를 분석기에 추가하는 데에 용이합니다. 
예를 들어 `사귀다` 동사의 과거형은 `사귀었다`가 맞지만, 흔히 `사겼다`로 잘못 쓰이기도 합니다.
`사겼다`가 `사귀/VV + 었/EP + 다/EF`로 바르게 분석되도록 하는데에 이 메소드를 사용할 수 있습니다.
```python
kiwi.add_pre_analyzed_word('사겼다', ['사귀/VV', '었/EP', '다/EF'], -3)
kiwi.add_pre_analyzed_word('사겼다', [('사귀', 'VV', 0, 2), ('었', 'EP', 1, 2), ('다', 'EF', 2, 3)], -3)
```

후자의 경우 분석 결과의 각 형태소가 원본 문자열에서 차지하는 위치를 정확하게 지정해줌으로써, 
Kiwi 분석 결과에서 해당 형태소의 분석 결과가 정확하게 나오도록 합니다.
        '''
        return super().add_pre_analyzed_word(form, analyzed, score)

    def add_rule(self,
        tag:str,
        replacer:Callable[[str], str],
        score:Optional[float] = 0.,
    ) -> List[str]:
        '''.. versionadded:: 0.11.0

규칙에 의해 변형된 형태소를 일괄적으로 추가합니다.

Parameters
----------
tag: str
    추가할 형태소들의 품사
replacer: Callable[[str], str]
    형태소를 변형시킬 규칙. 
    이 값은 호출가능한 Callable 형태로 제공되어야 하며, 원본 형태소 str를 입력으로 받아 변형된 형태소의 str을 반환해야합니다.
    만약 입력과 동일한 값을 반환하면 해당 변형 결과는 무시됩니다.
score: float
    추가할 변형된 형태소의 가중치 점수. 
    해당 형태에 부합하는 형태소 조합이 여러 개가 있는 경우, 이 점수가 높을 단어가 더 우선권을 가집니다.

Returns
-------
inserted_forms: List[str]
    규칙에 의해 새로 생성된 형태소의 `list`를 반환합니다.
        '''
        return super().add_rule(tag, replacer, score)
    
    def add_re_rule(self,
        tag:str,
        pattern:Union[str, 're.Pattern'],
        repl:Union[str, Callable],
        score:Optional[float] = 0.,
    ) -> List[str]:
        '''.. versionadded:: 0.11.0

`kiwipiepy.Kiwi.add_rule`과 동일한 역할을 수행하되, 변형 규칙에 정규표현식을 사용합니다.

Parameters
----------
tag: str
    추가할 형태소들의 품사
pattern: Union[str, re.Pattern]
    변형시킬 형태소의 규칙. 이 값은 `re.compile`로 컴파일가능한 정규표현식이어야 합니다.
repl: Union[str, Callable]
    `pattern`에 의해 발견된 패턴은 이 값으로 교체됩니다. `re.sub` 함수의 `repl` 인자와 동일합니다.
score: float
    추가할 변형된 형태소의 가중치 점수. 
    해당 형태에 부합하는 형태소 조합이 여러 개가 있는 경우, 이 점수가 높을 단어가 더 우선권을 가집니다.

Returns
-------
inserted_forms: List[str]
    규칙에 의해 새로 생성된 형태소의 `list`를 반환합니다.

Notes
-----
이 메소드는 규칙에 의해 변형되는 이형태들을 일괄적으로 추가할 때 굉장히 용이합니다.
예를 들어 `-요`가 `-염`으로 교체된 종결어미들(`먹어염`, `뛰었구염`, `배불러염` 등)을 일괄 등록하기 위해서는
다음을 수행하면 됩니다.

```python
kiwi.add_re_rule('EF', r'요$', r'염', -3.0)
```

이런 이형태들을 대량으로 등록할 경우 이형태가 원본 형태보다 분석결과에서 높은 우선권을 가지지 않도록
score를 `-3` 이하의 값으로 설정하는걸 권장합니다.
        '''
        if isinstance(pattern, str):
            pattern = re.compile(pattern)
        return super().add_rule(tag, lambda x:pattern.sub(repl, x), score)

    def load_user_dictionary(self,
        dict_path:str
    ) -> int:
        '''사용자 정의 사전을 읽어옵니다. 사용자 정의 사전 파일은 UTF-8로 인코딩된 텍스트 파일이어야 합니다.
사용자 정의 사전으로 추가된 단어의 개수를 반환합니다.
    
Parameters
----------
dict_path: str
    사용자 정의 사전 파일의 경로

Returns
-------
added_cnt: int
    사용자 정의 사전 파일을 이용해 추가된 단어의 개수를 반환합니다.

Notes
-----
사용자 정의 사전 파일의 형식에 대해서는 <a href='#_3'>여기</a>를 참조하세요.
        '''

        return super().load_user_dictionary(dict_path)

    def extract_words(self,
        texts,
        min_cnt:Optional[int] = 10,
        max_word_len:Optional[int] = 10,
        min_score:Optional[float] = 0.25,
        pos_score:Optional[float] = -3.,
        lm_filter:Optional[bool] = True,
    ):
        '''말뭉치로부터 새로운 단어를 추출합니다. 
이 기능은 https://github.com/lovit/soynlp 의 Word Extraction 기법을 바탕으로 하되, 
문자열 기반의 확률 모델을 추가하여 명사일 것으로 예측되는 단어만 추출합니다.

.. versionchanged:: 0.10.0
    이 메소드는 0.10.0 버전에서 사용법이 일부 변경되었습니다. 자세한 내용은 <a href="#0100">여기</a>를 확인해주세요.

Parameters
----------
texts: Iterable[str]
    분석할 문자열의 리스트, 혹은 Iterable입니다.
min_cnt: int
    추출할 단어의 최소 출현 빈도입니다. 이 빈도보다 적게 등장한 문자열은 단어 후보에서 제외됩니다.
max_word_len: int
    추출할 단어 후보의 최대 길이입니다. 이 길이보다 긴 단어 후보는 탐색되지 않습니다.
min_score: float
    단어 후보의 최소 점수입니다. 이 점수보다 낮은 단어 후보는 고려되지 않습니다.
    이 값을 낮출수록 단어가 아닌 형태가 추출될 가능성이 높아지고, 반대로 이 값을 높일 수록 추출되는 단어의 개수가 줄어들므로 적절한 수치로 설정할 필요가 있습니다.
pos_score: float
    ..versionadded:: 0.10.0

    단어 후보의 품사 점수입니다. 품사 점수가 이 값보다 낮은 경우 후보에서 제외됩니다.

lm_filter: bool
    ..versionadded:: 0.10.0
    
    True일 경우 품사 점수 및 언어 모델을 이용한 필터링을 수행합니다.
Returns
-------
result: List[Tuple[str, float, int, float]]
    추출된 단어후보의 목록을 반환합니다. 리스트의 각 항목은 (단어 형태, 최종 점수, 출현 빈도, 품사 점수)로 구성된 튜플입니다.
        '''

        return super().extract_words(
            texts,
            min_cnt,
            max_word_len,
            min_score,
            pos_score,
            lm_filter,
        )
    
    def extract_filter_words(self, *args, **kwargs):
        '''.. deprecated:: 0.10.0
    이 메소드의 기능은 `kiwipiepy.Kiwi.extract_words`로 통합되었습니다. 
    현재 이 메소드를 호출하는 것은 `kiwipiepy.Kiwi.extract_words`를 호출하는 것과 동일하게 처리됩니다.
        '''

        warnings.warn(
            "`extract_filter_words` has same effect to `extract_words` and will be removed in future version.",
            DeprecationWarning
        )
        return self.extract_words(*args, **kwargs)
    
    def extract_add_words(self,
        texts,
        min_cnt:Optional[int] = 10,
        max_word_len:Optional[int] = 10,
        min_score:Optional[float] = 0.25,
        pos_score:Optional[float] = -3.,
        lm_filter:Optional[bool] = True,
    ):
        '''말뭉치로부터 새로운 단어를 추출하고 새로운 명사에 적합한 결과들만 추려냅니다. 그리고 그 결과를 현재 모델에 자동으로 추가합니다.

.. versionchanged:: 0.10.0
    이 메소드는 0.10.0 버전에서 사용법이 일부 변경되었습니다. 자세한 내용은 <a href="#0100">여기</a>를 확인해주세요.

Parameters
----------
texts: Iterable[str]
    분석할 문자열의 리스트, 혹은 Iterable입니다.
min_cnt: int
    추출할 단어의 최소 출현 빈도입니다. 이 빈도보다 적게 등장한 문자열은 단어 후보에서 제외됩니다.
max_word_len: int
    추출할 단어 후보의 최대 길이입니다. 이 길이보다 긴 단어 후보는 탐색되지 않습니다.
min_score: float
    단어 후보의 최소 점수입니다. 이 점수보다 낮은 단어 후보는 고려되지 않습니다.
    이 값을 낮출수록 단어가 아닌 형태가 추출될 가능성이 높아지고, 반대로 이 값을 높일 수록 추출되는 단어의 개수가 줄어들므로 적절한 수치로 설정할 필요가 있습니다.
pos_score: float
    단어 후보의 품사 점수입니다. 품사 점수가 이 값보다 낮은 경우 후보에서 제외됩니다.
lm_filter: bool
    ..versionadded:: 0.10.0

    True일 경우 품사 점수 및 언어 모델을 이용한 필터링을 수행합니다.

Returns
-------
result: List[Tuple[str, float, int, float]]
    추출된 단어후보의 목록을 반환합니다. 리스트의 각 항목은 (단어 형태, 최종 점수, 출현 빈도, 품사 점수)로 구성된 튜플입니다.
        '''

        return super().extract_add_words(
            texts,
            min_cnt,
            max_word_len,
            min_score,
            pos_score,
            lm_filter,
        )
    
    def perform(self,
        texts,
        top_n:Optional[int] = 1,
        match_options:Optional[int] = Match.ALL,
        min_cnt:Optional[int] = 10,
        max_word_len:Optional[int] = 10,
        min_score:Optional[float] = 0.25,
        pos_score:Optional[float] = -3.,
        lm_filter:Optional[bool] = True,
    ):
        '''현재 모델의 사본을 만들어
`kiwipiepy.Kiwi.extract_add_words`메소드로 말뭉치에서 단어를 추출하여 추가하고, `kiwipiepy.Kiwi.analyze`로 형태소 분석을 실시합니다.
이 메소드 호출 후 모델의 사본은 파괴되므로, 말뭉치에서 추출된 단어들은 다시 모델에서 제거되고, 메소드 실행 전과 동일한 상태로 돌아갑니다.

.. versionchanged:: 0.10.0
    입력을 단순히 문자열의 리스트로 주고, 분석 결과 역시 별도의 `receiver`로 받지 않고 바로 메소드의 리턴값으로 받게 변경되었습니다.
    자세한 내용은 <a href="#0100">여기</a>를 확인해주세요.

.. deprecated:: 0.10.1
    추후 버전에서 변경, 혹은 제거될 가능성이 있는 메소드입니다.

Parameters
----------
texts: Iterable[str]
    분석할 문자열의 리스트, 혹은 Iterable입니다.
top_n: int
    분석 결과 후보를 상위 몇 개까지 생성할 지 설정합니다.
match_options: kiwipiepy.const.Match
    .. versionadded:: 0.8.0

    추출한 특수 문자열 패턴을 지정합니다. `kiwipiepy.const.Match`의 조합으로 설정할 수 있습니다.
min_cnt: int
    추출할 단어의 최소 출현 빈도입니다. 이 빈도보다 적게 등장한 문자열은 단어 후보에서 제외됩니다.
max_word_len: int
    추출할 단어 후보의 최대 길이입니다. 이 길이보다 긴 단어 후보는 탐색되지 않습니다.
min_score: float
    단어 후보의 최소 점수입니다. 이 점수보다 낮은 단어 후보는 고려되지 않습니다.
pos_score: float
    단어 후보의 품사 점수입니다. 품사 점수가 이 값보다 낮은 경우 후보에서 제외됩니다.

Returns
-------
results: Iterable[List[Tuple[List[kiwipiepy.Token], float]]]
    반환값은 `kiwipiepy.Kiwi.analyze`의 results와 동일합니다.
        '''

        warnings.warn(
            "`perform()` will be removed in future version.",
            DeprecationWarning
        )
        return super().perform(
            texts,
            top_n,
            match_options,
            min_cnt,
            max_word_len,
            min_score,
            pos_score,
            lm_filter,
        )
    
    def set_cutoff_threshold(self,
        threshold:float
    ):
        '''Beam 탐색 시 미리 제거할 후보의 점수 차를 설정합니다. 이 값이 클 수록 더 많은 후보를 탐색하게 되므로 분석 속도가 느려지지만 정확도가 올라갑니다.
반대로 이 값을 낮추면 더 적은 후보를 탐색하여 속도가 빨라지지만 정확도는 낮아집니다. 초기값은 5입니다.

.. versionadded:: 0.9.0
    초기값이 8에서 5로 변경되었습니다.

.. deprecated:: 0.10.0
    차기 버전에서 제거될 예정입니다.
    이 메소드 대신 `Kiwi.cutoff_threshold`를 직접 수정하십시오.

Parameters
----------
threshold: float
    0 보다 큰 실수
        '''

        warnings.warn(
            "`set_cutoff_threshold(v)` will be removed in future version. Use `Kiwi.cutoff_threshold = v` instead.",
            DeprecationWarning
        )
        self._cutoff_threshold = threshold
    
    def prepare(self):
        '''.. deprecated:: 0.10.0
    0.10.0버전부터 내부적으로 prepare()가 필요한 순간에 스스로 처리를 하도록 변경되어서 이제 이 메소드를 직접 호출할 필요가 없습니다.
    차기 버전에서 제거될 예정입니다.
        '''

        warnings.warn(
            "`prepare()` has no effect and will be removed in future version.",
            DeprecationWarning
        )
    
    def analyze(self,
        text:Union[str, Iterable[str]],
        top_n:Optional[int] = 1,
        match_options:Optional[int] = Match.ALL,
        normalize_coda:Optional[bool] = False
    ):
        '''형태소 분석을 실시합니다.

.. versionchanged:: 0.10.0
    이 메소드는 0.10.0 버전에서 사용법이 일부 변경되었습니다. 자세한 내용은 <a href="#0100">여기</a>를 확인해주세요.

Parameters
----------
text: Union[str, Iterable[str]]
    분석할 문자열입니다. 
    이 인자를 단일 str로 줄 경우, 싱글스레드에서 처리하며
    str의 Iterable로 줄 경우, 멀티스레드로 분배하여 처리합니다.
top_n: int
    분석 결과 후보를 상위 몇 개까지 생성할 지 설정합니다.
match_options: kiwipiepy.const.Match
    
    .. versionadded:: 0.8.0
    
    추출한 특수 문자열 패턴을 지정합니다. `kiwipiepy.const.Match`의 조합으로 설정할 수 있습니다.
normalize_coda: bool

    .. versionadded:: 0.10.2

    True인 경우 '먹었엌ㅋㅋ'처럼 받침이 덧붙어서 분석에 실패하는 경우, 받침을 분리하여 정규화합니다.

Returns
-------
result: List[Tuple[List[kiwipiepy.Token], float]]
    text를 str으로 준 경우.
    분석 결과는 최대 `top_n`개 길이의 리스트로 반환됩니다. 리스트의 각 항목은 `(분석 결과, 분석 점수)`로 구성된 튜플입니다. 
    `분석 결과`는 `kiwipiepy.Token`의 리스트로 구성됩니다.

results: Iterable[List[Tuple[List[kiwipiepy.Token], float]]]
    text를 Iterable[str]으로 준 경우.
    반환값은 iterator로 주어집니다. iterator가 차례로 반환하는 분석결과 값은 입력으로 준 text의 순서와 동일합니다.

Notes
-----
`text`를 `Iterable`로 준 경우 멀티스레딩으로 처리됩니다. 
이 때 사용되는 스레드의 개수는 처음에 `Kiwi`를 생성할 때 `num_workers`로 준 값입니다.

```python
kiwi.analyze('형태소 분석 결과입니다')
# 반환 값은 다음과 같이 `(형태소 분석 결과, 분석 점수)`의 형태입니다.
(
    [Token(form='형태소', tag='NNG', start=0, len=3), 
    Token(form='분석', tag='NNG', start=4, len=2), 
    Token(form='결과', tag='NNG', start=7, len=2), 
    Token(form='이', tag='VCP', start=9, len=1), 
    Token(form='ᆸ니다', tag='EF', start=10, len=2)
    ],
    -34.3332
)

# 4개의 스레드에서 동시에 처리합니다.
kiwi = Kiwi(num_workers=4)
with open('result.txt', 'w', encoding='utf-8') as output:
    for res in kiwi.analyze(open('test.txt', encoding='utf-8')):
        print(' '.join(map(lambda x:x[0]+'/'+x[1], res[0][0])), file=output)
```
        '''
        if normalize_coda:
            match_options |= Match.NORMALIZING_CODA
        return super().analyze(text, top_n, match_options)
    
    def get_option(self,
        option:int,
    ):
        '''현재 모델의 설정값을 가져옵니다.

.. deprecated:: 0.10.0
    차기 버전에서 제거될 예정입니다. 
    이 메소드 대신 `Kiwi.integrate_allomorph`값을 직접 읽으십시오.

Parameters
----------
option: kiwipiepy.const.Option
    검사할 옵션의 열거값. 현재는 `kiwipiepy.const.Option.INTEGRATE_ALLOMORPH`만 지원합니다.

Returns
-------
value: int
    해당 옵션이 설정되어 있는 경우 1, 아닌 경우 0을 반환합니다.
        '''

        warnings.warn(
            "`get_option()` will be removed in future version.",
            DeprecationWarning
        )
        if option != Option.INTEGRATE_ALLOMORPH: raise ValueError("Wrong `option` value: {}".format(option))
        return int(self._integrate_allomorph)
    
    def set_option(self, 
        option:int,
        value:int,
    ):
        '''현재 모델의 설정값을 변경합니다.

.. deprecated:: 0.10.0
    차기 버전에서 제거될 예정입니다. 
    이 메소드 대신 `Kiwi.integrate_allomorph`값을 직접 수정하십시오.

Parameters
----------
option: kiwipiepy.const.Option
    변경할 옵션의 열거값. 현재는 `kiwipiepy.const.Option.INTEGRATE_ALLOMORPH`만 지원합니다.
value: int
    0으로 설정할 경우 해당 옵션을 해제, 0이 아닌 값으로 설정할 경우 해당 옵션을 설정합니다.
        '''

        warnings.warn(
            "`set_option()` will be removed in future version.",
            DeprecationWarning
        )
        if option != Option.INTEGRATE_ALLOMORPH: raise ValueError("Wrong `option` value: {}".format(option))
        self._integrate_allomorph = bool(value)
    
    def morpheme(self,
        idx:int,
    ):
        return super().morpheme(idx)
    
    @property
    def version(self):
        '''Kiwi의 버전을 반환합니다. 
.. deprecated:: 0.10.0
    차기 버전에서 제거될 예정입니다. 대신 `kiwipiepy.__version__`을 사용하십시오.
        '''

        return __version__
    
    def _on_build(self):
        self._integrate_allomorph = self._ns_integrate_allomorph
        self._cutoff_threshold = self._ns_cutoff_threshold
        self._unk_form_score_scale = self._ns_unk_form_score_scale
        self._unk_form_score_bias = self._ns_unk_form_score_bias
        self._space_penalty = self._ns_space_penalty
        self._max_unk_form_size = self._ns_max_unk_form_size
        self._space_tolerance = self._ns_space_tolerance
        self._typo_cost_weight = self._ns_typo_cost_weight

    @property
    def cutoff_threshold(self):
        '''.. versionadded:: 0.10.0

Beam 탐색 시 미리 제거할 후보의 점수 차를 설정합니다. 이 값이 클 수록 더 많은 후보를 탐색하게 되므로 분석 속도가 느려지지만 정확도가 올라갑니다.
반대로 이 값을 낮추면 더 적은 후보를 탐색하여 속도가 빨라지지만 정확도는 낮아집니다. 초기값은 5입니다.
        '''

        return self._ns_cutoff_threshold
    
    @cutoff_threshold.setter
    def cutoff_threshold(self, v:float):
        self._cutoff_threshold = self._ns_cutoff_threshold = float(v)
    
    @property
    def integrate_allomorph(self):
        '''.. versionadded:: 0.10.0

True일 경우 음운론적 이형태를 통합하여 출력합니다. /아/와 /어/나 /았/과 /었/ 같이 앞 모음의 양성/음성에 따라 형태가 바뀌는 어미들을 하나로 통합하여 출력합니다.
        '''

        return self._ns_integrate_allomorph
    
    @integrate_allomorph.setter
    def integrate_allomorph(self, v:bool):
        self._integrate_allomorph = self._ns_integrate_allomorph = bool(v)
    
    @property
    def space_penalty(self):
        '''.. versionadded:: 0.11.1

형태소 중간에 삽입된 공백 문자가 있을 경우 언어모델 점수에 추가하는 페널티 점수입니다. 기본값은 7.0입니다.
        '''

        return self._ns_space_penalty
    
    @space_penalty.setter
    def space_penalty(self, v:float):
        self._space_penalty = self._ns_space_penalty = float(v)

    @property
    def space_tolerance(self):
        '''.. versionadded:: 0.11.1

형태소 중간에 삽입된 공백문자를 몇 개까지 허용할지 설정합니다. 기본값은 0이며, 이 경우 형태소 중간에 공백문자가 삽입되는 걸 허용하지 않습니다.

`Kiwi.space` 메소드 참고.
        '''

        return self._ns_space_tolerance
    
    @space_tolerance.setter
    def space_tolerance(self, v:int):
        if v < 0: raise ValueError("`space_tolerance` must be a zero or positive integer.")
        self._space_tolerance = self._ns_space_tolerance = int(v)

    @property
    def max_unk_form_size(self):
        '''.. versionadded:: 0.11.1

분석 과정에서 허용할 미등재 형태의 최대 길이입니다. 기본값은 6입니다.
        '''

        return self._ns_max_unk_form_size
    
    @max_unk_form_size.setter
    def max_unk_form_size(self, v:int):
        if v < 0: raise ValueError("`max_unk_form_size` must be a zero or positive integer.")
        self._max_unk_form_size = self._ns_max_unk_form_size = int(v)

    @property
    def typo_cost_weight(self):
        '''.. versionadded:: 0.13.0

오타 교정 시에 사용할 교정 가중치. 이 값이 클수록 교정을 보수적으로 수행합니다. 기본값은 6입니다.
        '''

        return self._ns_typo_cost_weight
    
    @typo_cost_weight.setter
    def typo_cost_weight(self, v:float):
        if v < 0: raise ValueError("`typo_cost_weight` must be a zero or positive float.")
        self._typo_cost_weight = self._ns_typo_cost_weight = float(v)

    @property
    def num_workers(self):
        '''.. versionadded:: 0.10.0

병렬처리시 사용할 스레드의 개수입니다. (읽기 전용)
        '''
        
        return self._num_workers
    
    @property
    def model_type(self):
        '''.. versionadded:: 0.13.0

형태소 분석에 사용 중인 언어 모델의 종류 (읽기 전용)
        '''
        return self._ns_model_type

    def _tokenize(self, 
        text:Union[str, Iterable[str]], 
        match_options:Optional[int] = Match.ALL,
        normalize_coda:Optional[bool] = False,
        split_sents:Optional[bool] = False,
        stopwords:Optional[Stopwords] = None,
        echo:Optional[bool] = False,
    ):
        import itertools

        def _refine_result(results):
            if not split_sents:
                return results[0][0] if stopwords is None else stopwords.filter(results[0][0])
            
            tokens, _ = results[0]
            ret = [list(g) if stopwords is None else stopwords.filter(g) for k, g in itertools.groupby(tokens, key=lambda x:x.sent_position)]
            return ret
        
        def _refine_result_with_echo(arg):
            results, raw_input = arg
            return _refine_result(results), raw_input

        if normalize_coda:
            match_options |= Match.NORMALIZING_CODA

        if isinstance(text, str):
            echo = False
            return _refine_result(super().analyze(text, top_n=1, match_options=match_options))
            
        return map(_refine_result_with_echo if echo else _refine_result, super().analyze(text, top_n=1, match_options=match_options, echo=echo))

    def tokenize(self, 
        text:Union[str, Iterable[str]], 
        match_options:Optional[int] = Match.ALL,
        normalize_coda:Optional[bool] = False,
        split_sents:Optional[bool] = False,
        stopwords:Optional[Stopwords] = None,
        echo:Optional[bool] = False,
    ):
        '''.. versionadded:: 0.10.2

`analyze`와는 다르게 형태소 분석결과만 간단하게 반환합니다.

Parameters
----------
text: Union[str, Iterable[str]]
    분석할 문자열입니다. 
    이 인자를 단일 str로 줄 경우, 싱글스레드에서 처리하며
    str의 Iterable로 줄 경우, 멀티스레드로 분배하여 처리합니다.
match_options: kiwipiepy.const.Match
    추출한 특수 문자열 패턴을 지정합니다. `kiwipiepy.const.Match`의 조합으로 설정할 수 있습니다.
normalize_coda: bool
    True인 경우 '먹었엌ㅋㅋ'처럼 받침이 덧붙어서 분석에 실패하는 경우, 받침을 분리하여 정규화합니다.
split_sents: bool
    .. versionadded:: 0.10.3

    True인 경우 형태소 리스트를 문장별로 묶어서 반환합니다. 자세한 내용은 아래의 Returns 항목을 참조하세요.
stopwords: Stopwords
    .. versionadded:: 0.10.3

    이 인자로 None이 아닌 `kiwipiepy.utils.Stopwords` 객체를 줄 경우, 형태소 분석 결과 중 그 객체에 포함되는 불용어를 제외한 나머지 결과만을 반환합니다.
echo: bool
    .. versionadded:: 0.11.2

    이 값이 True이고 `text`를 str의 Iterable로 준 경우, 분석 결과뿐만 아니라 원본 입력을 함께 반환합니다. `text`가 단일 str인 경우 이 인자는 무시됩니다.

Returns
-------
result: List[kiwipiepy.Token]
    split_sents == False일때 text를 str으로 준 경우.
    `kiwipiepy.Token`의 리스트를 반환합니다.

results: Iterable[List[kiwipiepy.Token]]
    split_sents == False일때 text를 Iterable[str]으로 준 경우.
    반환값은 `result`의 iterator로 주어집니다. iterator가 차례로 반환하는 분석결과 값은 입력으로 준 text의 순서와 동일합니다.

results_with_echo: Iterable[Tuple[List[kiwipiepy.Token], str]]
    split_sents == False이고 echo=True일때 text를 Iterable[str]으로 준 경우.
    반환값은 (`result`의 iterator, `raw_input`)으로 주어집니다. iterator가 차례로 반환하는 분석결과 값은 입력으로 준 text의 순서와 동일합니다.

result_by_sent: List[List[kiwipiepy.Token]]
    split_sents == True일때 text를 str으로 준 경우.
    형태소 분석 결과가 문장별로 묶여서 반환됩니다.
    즉, 전체 문장이 n개라고 할 때, `result_by_sent[0] ~ result_by_sent[n-1]`에는 각 문장별 분석 결과가 들어갑니다.

results_by_sent: Iterable[List[List[kiwipiepy.Token]]]
    split_sents == True일때 text를 Iterable[str]으로 준 경우.
    반환값은 `result_by_sent`의 iterator로 주어집니다. iterator가 차례로 반환하는 분석결과 값은 입력으로 준 text의 순서와 동일합니다.

results_by_sent_with_echo: Iterable[Tuple[List[List[kiwipiepy.Token]], str]]
    split_sents == True이고 echo=True일때 text를 Iterable[str]으로 준 경우.
    반환값은 (`result_by_sent`의 iterator, `raw_input`)으로 주어집니다. iterator가 차례로 반환하는 분석결과 값은 입력으로 준 text의 순서와 동일합니다.

Notes
-----

```python
>> kiwi.tokenize("안녕하세요 형태소 분석기 키위입니다.")
[Token(form='안녕', tag='NNG', start=0, len=2),
 Token(form='하', tag='XSA', start=2, len=1),
 Token(form='시', tag='EP', start=4, len=1),
 Token(form='어요', tag='EC', start=3, len=2),
 Token(form='형태소', tag='NNG', start=6, len=3),
 Token(form='분석', tag='NNG', start=10, len=2),
 Token(form='기', tag='NNG', start=12, len=1),
 Token(form='키위', tag='NNG', start=14, len=2),
 Token(form='이', tag='VCP', start=16, len=1),
 Token(form='ᆸ니다', tag='EF', start=17, len=2),
 Token(form='.', tag='SF', start=19, len=1)]

# normalize_coda 옵션을 사용하면 
# 덧붙은 받침 때문에 분석이 깨지는 경우를 방지할 수 있습니다.
>> kiwi.tokenize("ㅋㅋㅋ 이런 것도 분석이 될까욬ㅋㅋ?", normalize_coda=True)
[Token(form='ㅋㅋㅋ', tag='SW', start=0, len=3),
 Token(form='이런', tag='MM', start=4, len=2),
 Token(form='것', tag='NNB', start=7, len=1),
 Token(form='도', tag='JX', start=8, len=1),
 Token(form='분석', tag='NNG', start=10, len=2),
 Token(form='이', tag='JKS', start=12, len=1),
 Token(form='되', tag='VV', start=14, len=1),
 Token(form='ᆯ까요', tag='EC', start=15, len=2),
 Token(form='ㅋㅋㅋ', tag='SW', start=17, len=2),
 Token(form='?', tag='SF', start=19, len=1)]

# Stopwords 클래스를 바로 적용하여 불용어를 걸러낼 수 있습니다.
>> from kiwipiepy.utils import Stopwords
>> stopwords = Stopwords()
>> kiwi.tokenize("분석 결과에서 불용어만 제외하고 출력할 수도 있다.", stopwords=stopwords)
[Token(form='분석', tag='NNG', start=0, len=2),
 Token(form='결과', tag='NNG', start=3, len=2),
 Token(form='불', tag='XPN', start=8, len=1),
 Token(form='용어', tag='NNG', start=9, len=2),
 Token(form='제외', tag='NNG', start=13, len=2),
 Token(form='출력', tag='NNG', start=18, len=2)]
```
        '''
        return self._tokenize(text, match_options, normalize_coda, split_sents, stopwords, echo)

    def split_into_sents(self, 
        text:Union[str, Iterable[str]], 
        match_options:Optional[int] = Match.ALL, 
        normalize_coda:Optional[bool] = False,
        return_tokens:Optional[bool] = False,
        return_sub_sents:Optional[bool] = True,
    ) -> Union[List[Sentence], Iterable[List[Sentence]]]:
        '''..versionadded:: 0.10.3

입력 텍스트를 문장 단위로 분할하여 반환합니다. 
이 메소드는 문장 분할 과정에서 내부적으로 형태소 분석을 사용하므로 문장 분할과 동시에 형태소 분석 결과를 얻는 데 사용할 수도 있습니다.

Parameters
----------
text: Union[str, Iterable[str]]
    분석할 문자열입니다. 
    이 인자를 단일 str로 줄 경우, 싱글스레드에서 처리하며
    str의 Iterable로 줄 경우, 멀티스레드로 분배하여 처리합니다.
match_options: kiwipiepy.const.Match
    추출한 특수 문자열 패턴을 지정합니다. `kiwipiepy.const.Match`의 조합으로 설정할 수 있습니다.
normalize_coda: bool
    True인 경우 '먹었엌ㅋㅋ'처럼 받침이 덧붙어서 분석에 실패하는 경우, 받침을 분리하여 정규화합니다.
return_tokens: bool
    True인 경우 문장별 형태소 분석 결과도 함께 반환합니다.
return_sub_Sents: bool
    ..versionadded:: 0.14.0

    True인 경우 문장 내 안긴 문장의 목록도 함께 반환합니다.

Returns
-------
sentences: List[kiwipiepy.Sentence]
    text를 str으로 준 경우.
    문장 분할 결과인 `kiwipiepy.Sentence`의 리스트를 반환합니다.

iterable_of_sentences: Iterable[List[kiwipiepy.Sentence]]
    text를 Iterable[str]으로 준 경우.
    반환값은 `sentences`의 iterator로 주어집니다. iterator가 차례로 반환하는 분석결과 값은 입력으로 준 text의 순서와 동일합니다.

Notes
-----
문장 분리 기능은 형태소 분석에 기반합니다. 따라서 return_tokens의 True/False여부에 상관없이 내부적으로 형태소 분석을 수행하며,
출력 시에만 형태소 목록을 넣거나 뺄 뿐이므로 이에 따른 실행 속도 차이는 없습니다.
그러므로 문장 분리 기능과 형태소 분석을 동시에 수행해야하는 경우 `return_tokens=True`로 설정하는 게, 
문장 분리 후 따로 형태소 분석을 수행하는 것보다 효율적입니다.

```python
>> kiwi.split_into_sents("여러 문장으로 구성된 텍스트네 이걸 분리해줘")
[Sentence(text='여러 문장으로 구성된 텍스트네', start=0, end=16, tokens=None, subs=[]),
 Sentence(text='이걸 분리해줘', start=17, end=24, tokens=None, subs=[])]

>> kiwi.split_into_sents("여러 문장으로 구성된 텍스트네 이걸 분리해줘", return_tokens=True)
[Sentence(text='여러 문장으로 구성된 텍스트네', start=0, end=16, tokens=[
  Token(form='여러', tag='MM', start=0, len=2), 
  Token(form='문장', tag='NNG', start=3, len=2), 
  Token(form='으로', tag='JKB', start=5, len=2), 
  Token(form='구성', tag='NNG', start=8, len=2), 
  Token(form='되', tag='XSV', start=10, len=1), 
  Token(form='ᆫ', tag='ETM', start=10, len=1), 
  Token(form='텍스트', tag='NNG', start=12, len=3), 
  Token(form='이', tag='VCP', start=15, len=1), 
  Token(form='네', tag='EF', start=15, len=1)
 ], subs=[]),
 Sentence(text='이걸 분리해줘', start=17, end=24, tokens=[
  Token(form='이거', tag='NP', start=17, len=2), 
  Token(form='ᆯ', tag='JKO', start=19, len=0), 
  Token(form='분리', tag='NNG', start=20, len=2), 
  Token(form='하', tag='XSV', start=22, len=1), 
  Token(form='어', tag='EC', start=22, len=1), 
  Token(form='주', tag='VX', start=23, len=1), 
  Token(form='어', tag='EF', start=23, len=1)
 ], subs=[])]

# 0.14.0 버전부터는 문장 안에 또 다른 문장이 포함된 경우도 처리 가능
>> kiwi.split_into_sents("회사의 정보 서비스를 책임지고 있는 로웬버그John Loewenberg는" 
     "<서비스 산업에 있어 종이는 혈관내의 콜레스트롤과 같다. 나쁜 종이는 동맥을 막는 내부의 물질이다.>"
     "라고 말한다.")
[Sentence(text='회사의 정보 서비스를 책임지고 있는 로웬버그John Loewenberg는' 
  '<서비스 산업에 있어 종이는 혈관내의 콜레스트롤과 같다. 나쁜 종이는 동맥을 막는 내부의 물질이다.>'
  '라고 말한다.', start=0, end=104, tokens=None, subs=[
  Sentence(text='서비스 산업에 있어 종이는 혈관내의 콜레스트롤과 같다.', start=42, end=72, tokens=None, subs=None), 
  Sentence(text='나쁜 종이는 동맥을 막는 내부의 물질이다.', start=73, end=96, tokens=None, subs=None)
])]
```
        '''
        def _make_result(arg):
            sents, raw_input = arg
            ret = []
            for sent in sents:
                start = sent[0].start
                end = sent[-1].end
                tokens = sent if return_tokens else None
                subs = None
                if return_sub_sents:
                    subs = []
                    sub_toks = []
                    last = 0
                    for tok in sent:
                        if tok.sub_sent_position != last:
                            if last:
                                subs.append(Sentence(raw_input[sub_start:last_end], sub_start, last_end, sub_toks if return_tokens else None, None))
                                sub_toks = []
                            sub_start = tok.start
                        if tok.sub_sent_position:
                            sub_toks.append(tok)
                        last = tok.sub_sent_position
                        last_end = tok.end
                ret.append(Sentence(raw_input[start:end], start, end, tokens, subs))
            return ret

        if isinstance(text, str):
            return _make_result((self._tokenize(text, match_options=match_options, normalize_coda=normalize_coda, split_sents=True), text))

        return map(_make_result, self._tokenize(text, match_options=match_options, normalize_coda=normalize_coda, split_sents=True, echo=True))

    def glue(self,
        text_chunks:Iterable[str],
        return_space_insertions:Optional[bool] = False,
    ) -> Union[str, Tuple[str, List[bool]]]:
        '''..versionadded:: 0.11.1

여러 텍스트 조각을 하나로 합치되, 문맥을 고려해 적절한 공백을 사이에 삽입합니다.

Parameters
----------
text_chunks: Iterable[str]
    합칠 텍스트 조각들의 목록입니다.
return_space_insertions: bool
    True인 경우, 각 조각별 공백 삽입 유무를 `List[bool]`로 반환합니다.
    기본값은 False입니다.

Returns
-------
result: str
    입력 텍스트 조각의 결합결과를 반환합니다.

space_insertions: Iterable[str]
    이 값은 `return_space_insertions=True`인 경우에만 반환됩니다.
    공백 삽입 유무를 알려줍니다.

Notes
-----
이 메소드의 공백 자동 삽입 기능은 형태소 분석에 기반합니다. 

```python
>> kiwi.glue([
    "그러나  알고보니 그 봉",
    "지 안에 있던 것은 바로",
    "레몬이었던 것이다."])
"그러나  알고보니 그 봉지 안에 있던 것은 바로 레몬이었던 것이다."

>> kiwi.glue([
    "그러나  알고보니 그 봉",
    "지 안에 있던 것은 바로",
    "레몬이었던 것이다."], return_space_insertions=True)
("그러나  알고보니 그 봉지 안에 있던 것은 바로 레몬이었던 것이다.", [False, True])
```
        '''

        all_chunks = []
        def _zip_consequences(it):
            prev = next(it).strip()
            all_chunks.append(prev)
            for s in it:
                s = s.strip()
                yield prev + ' ' + s
                yield prev + s
                prev = s
                all_chunks.append(prev)
        
        riter = super().analyze(_zip_consequences(iter(text_chunks)), 1, Match.ALL)
        i = 0
        ret = []
        space_insertions = []
        try:
            while 1:
                _, score_with_space = next(riter)[0]
                _, score_without_space = next(riter)[0]
                ret.append(all_chunks[i])
                if score_with_space >= score_without_space or re.search(r'[0-9A-Za-z]$', all_chunks[i]):
                    ret.append(' ')
                    space_insertions.append(True)
                else:
                    space_insertions.append(False)
                i += 1
        except StopIteration:
            ret.append(all_chunks[i])
        
        if return_space_insertions:
            return ''.join(ret), space_insertions
        else:
            return ''.join(ret)

    def space(self,
        text:Union[str, Iterable[str]],
        reset_whitespace:Optional[bool] = False,
    ) -> Union[str, Iterable[str]]:
        '''..versionadded:: 0.11.1

입력 텍스트에서 띄어쓰기를 교정하여 반환합니다.

Parameters
----------
text: Union[str, Iterable[str]]
    분석할 문자열입니다. 
    이 인자를 단일 str로 줄 경우, 싱글스레드에서 처리하며
    str의 Iterable로 줄 경우, 멀티스레드로 분배하여 처리합니다.
reset_whitespace: bool
    True인 경우 이미 띄어쓰기된 부분을 붙이는 교정도 적극적으로 수행합니다. 
    기본값은 False로, 이 경우에는 붙어 있는 단어를 띄어쓰는 교정 위주로 수행합니다.

Returns
-------
result: str
    text를 str으로 준 경우.
    입력 텍스트의 띄어쓰기 교정 결과를 반환합니다.

iterable_of_results: Iterable[str]
    text를 Iterable[str]으로 준 경우.
    입력 텍스트의 띄어쓰기 교정 결과를 반환합니다. iterator가 차례로 반환하는 분석결과 값은 입력으로 준 text의 순서와 동일합니다.

Notes
-----
이 메소드의 띄어쓰기 교정 기능은 형태소 분석에 기반합니다. 
따라서 형태소 중간에 공백이 삽입된 경우 교정 결과가 부정확할 수 있습니다.
이 경우 `Kiwi.space_tolerance`를 조절하여 형태소 내 공백을 무시하거나, 
`reset_whitespace=True`로 설정하여 아예 기존 공백을 무시하고 띄어쓰기를 하도록 하면 결과를 개선할 수 있습니다.

```python
>> kiwi.space("띄어쓰기없이작성된텍스트네이걸교정해줘")
"띄어쓰기 없이 작성된 텍스트네 이걸 교정해 줘."
>> kiwi.space("띄 어 쓰 기 문 제 가 있 습 니 다")
"띄어 쓰기 문 제 가 있 습 니 다"
>> kiwi.space_tolerance = 2 # 형태소 내 공백을 최대 2개까지 허용
>> kiwi.space("띄 어 쓰 기 문 제 가 있 습 니 다")
"띄어 쓰기 문제가 있습니다"
>> kiwi.space("띄 어 쓰 기 문 제 가 있 습 니 다", reset_whitespace=True) # 기존 공백 전부 무시
"띄어쓰기 문제가 있습니다"
```
        '''
        ws = re.compile(r'(?<=[가-힣])\s+(?=[가-힣.,?!:;])')
        any_ws = re.compile(r'\s+')
        space_insertable = re.compile('|'.join([
            r'(([^SUWX]|X[RS]|S[EH]).* ([NMI]|V[VAX]|VCN|XR|XPN|S[WLHN]))',
            r'(SN ([MI]|N[PR]|NN[GP]|V[VAX]|VCN|XR|XPN|S[WH]))',
            r'((S[FPL]).* ([NMI]|V[VAX]|VCN|XR|XPN|S[WH]))',
        ]))

        def _reset(t):
            return ws.sub('', t)

        def _space(arg):
            tokens, raw = arg
            tokens = tokens[0][0]
            chunks = []
            last = 0
            prev_tag = None
            for t in tokens:
                if last < t.start:
                    if (t.tag.startswith('E') or t.tag.startswith('J') or t.tag.startswith('XS')
                        or t.tag == 'VX' and t.form in '하지'
                    ):
                        s = any_ws.sub('', raw[last:t.start])
                    else:
                        s = raw[last:t.start]
                    if s: chunks.append(s)
                    last = t.start
                if prev_tag and space_insertable.match(prev_tag + ' ' + t.tag):
                    if t.tag == 'VX' and t.form in '하지':
                        pass # 보조 용언 중 `하다/지다`는 붙여쓴다.
                    elif not chunks[-1][-1].isspace():
                        # 이전에 공백이 없는 경우만 삽입
                        chunks.append(' ') 
                if last < t.end:
                    s = any_ws.sub('', raw[last:t.end])
                    if s: chunks.append(s)
                last = t.end
                prev_tag = t.tag
            if last < len(raw):
                chunks.append(raw[last:])
            return ''.join(chunks)

        if isinstance(text, str):
            if reset_whitespace: text = _reset(text)
            return _space((super().analyze(text, 1, Match.ALL), text))
        else:
            if reset_whitespace: text = map(_reset, text)
            return map(_space, super().analyze(text, 1, Match.ALL, echo=True))

    def join(self, 
        morphs:Iterable[Tuple[str, str]],
        lm_search:Optional[bool] = True
    ) -> str:
        '''..versionadded:: 0.12.0

형태소들을 결합하여 문장으로 복원합니다. 
조사나 어미는 앞 형태소에 맞춰 적절한 형태로 변경됩니다.

Parameters
----------
morphs: Iterable[Union[Token, Tuple[str, str]]]
    결합할 형태소의 목록입니다. 
    각 형태소는 `Kiwi.tokenize`에서 얻어진 `Token` 타입이거나, 
    (형태, 품사)로 구성된 `tuple` 타입이어야 합니다.
lm_search: bool
    둘 이상의 형태로 복원 가능한 모호한 형태소가 있는 경우, 이 값이 True면 언어 모델 탐색을 통해 최적의 형태소를 선택합니다.
    False일 경우 탐색을 실시하지 않지만 더 빠른 속도로 복원이 가능합니다.

Returns
-------
result: str
    입력 형태소의 결합 결과를 반환합니다.

Notes
-----
`Kiwi.join`은 형태소를 결합할 때 `Kiwi.space`에서 사용하는 것과 유사한 규칙을 사용하여 공백을 적절히 삽입합니다.
형태소 그 자체에는 공백 관련 정보가 포함되지 않으므로
특정 텍스트를 `Kiwi.tokenize`로 분석 후 다시 `Kiwi.join`으로 결합하여도 원본 텍스트가 그대로 복원되지는 않습니다.


```python
>> kiwi.join([('덥', 'VA'), ('어', 'EC')])
'더워'
>> tokens = kiwi.tokenize("분석된결과를 다시합칠수있다!")
# 형태소 분석 결과를 복원. 
# 복원 시 공백은 규칙에 의해 삽입되므로 원문 텍스트가 그대로 복원되지는 않음.
>> kiwi.join(tokens)
'분석된 결과를 다시 합칠 수 있다!'
>> tokens[3]
Token(form='결과', tag='NNG', start=4, len=2)
>> tokens[3] = ('내용', 'NNG') # 4번째 형태소를 결과->내용으로 교체
>> kiwi.join(tokens) # 다시 join하면 결과를->내용을 로 교체된 걸 확인 가능
'분석된 내용을 다시 합칠 수 있다!'

# 불규칙 활용여부가 모호한 경우 lm_search=True인 경우 맥락을 고려해 최적의 후보를 선택합니다.
>> kiwi.join([('길', 'NNG'), ('을', 'JKO'), ('묻', 'VV'), ('어요', 'EF')])
'길을 물어요'
>> kiwi.join([('흙', 'NNG'), ('이', 'JKS'), ('묻', 'VV'), ('어요', 'EF')])
'흙이 묻어요'
# lm_search=False이면 탐색을 실시하지 않습니다.
>> kiwi.join([('길', 'NNG'), ('을', 'JKO'), ('묻', 'VV'), ('어요', 'EF')], lm_search=False)
'길을 묻어요'
>> kiwi.join([('흙', 'NNG'), ('이', 'JKS'), ('묻', 'VV'), ('어요', 'EF')], lm_search=False)
'흙이 묻어요'
# 동사/형용사 품사 태그 뒤에 -R(규칙 활용), -I(불규칙 활용)을 덧붙여 활용법을 직접 명시할 수 있습니다.
>> kiwi.join([('묻', 'VV-R'), ('어요', 'EF')])
'묻어요'
>> kiwi.join([('묻', 'VV-I'), ('어요', 'EF')])
'물어요'

# 과거형 선어말어미를 제거하는 예시
>> remove_past = lambda s: kiwi.join(t for t in kiwi.tokenize(s) if t.tagged_form != '었/EP')
>> remove_past('먹었다')
'먹다'
>> remove_past('먼 길을 걸었다')
'먼 길을 걷다'
>> remove_past('전화를 걸었다.')
'전화를 걸다.'
```
        '''
        return super().join(morphs, lm_search)

    def evaluate(self,
        sequences:List[List[int]],
        prefix:Optional[List[int]] = None,
        suffix:Optional[List[int]] = None,
    ):
        raise NotImplementedError

    def predict_next(self,
        prefix:List[int],
    ):
        raise NotImplementedError
