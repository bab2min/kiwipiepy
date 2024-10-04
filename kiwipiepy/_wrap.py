import re
from functools import partial
from typing import Callable, List, Dict, Optional, Tuple, Union, Iterable, NamedTuple, NewType, Any
from dataclasses import dataclass
import itertools
import warnings

import _kiwipiepy
from _kiwipiepy import _Kiwi, _TypoTransformer, _HSDataset, _MorphemeSet, _NgramExtractor
from kiwipiepy._c_api import Token
from kiwipiepy._version import __version__
from kiwipiepy.utils import Stopwords
from kiwipiepy.const import Match
from kiwipiepy.template import Template

Sentence = NamedTuple('Sentence', [('text', str), ('start', int), ('end', int), ('tokens', Optional[List[Token]]), ('subs', Optional[List['Sentence']])])
Sentence.__doc__ = '문장 분할 결과를 담기 위한 `namedtuple`입니다.'
Sentence.text.__doc__ = '분할된 문장의 텍스트'
Sentence.start.__doc__ = '전체 텍스트 내에서 분할된 문장이 시작하는 위치 (문자 단위)'
Sentence.end.__doc__ = '전체 텍스트 내에서 분할된 문장이 끝나는 위치 (문자 단위)'
Sentence.tokens.__doc__ = '분할된 문장의 형태소 분석 결과'
Sentence.subs.__doc__ = '''.. versionadded:: 0.14.0

현 문장 내에 포함된 안긴 문장의 목록
'''

POSTag = NewType('POSTag', str)
PretokenizedToken = NamedTuple('PretokenizedToken', [('form', str), ('tag', POSTag), ('start', int), ('end', int)])
PretokenizedToken.__doc__ = '''미리 분석된 형태소를 나타내는 데 사용하는 `namedtuple`입니다.'''
PretokenizedToken.form.__doc__ = '형태소의 형태'
PretokenizedToken.tag.__doc__ = '형태소의 품사 태그'
PretokenizedToken.start.__doc__ = '주어진 구간에서 형태소가 시작하는 시작 위치 (문자 단위)'
PretokenizedToken.end.__doc__ = '주어진 구간에서 형태소가 끝나는 시작 위치 (문자 단위)'
PretokenizedTokenList = List[Union[Tuple[int, int], Tuple[int, int, POSTag], Tuple[int, int, PretokenizedToken], Tuple[int, int, List[PretokenizedToken]]]]

NgramCandidate = NamedTuple('NgramCandidate', [('text', str), ('tokens', List[Tuple[str, str]]), ('token_scores', List[float]), ('cnt', int), ('df', int), ('score', float), ('npmi', float), ('lb_entropy', float), ('rb_entropy', float), ('lm_score', float)])

class NgramExtractor(_NgramExtractor):
    def __init__(self, kiwi, gather_lm_score=True):
        super().__init__(kiwi, gather_lm_score)

    def add(self, text:Union[str, List[str]]) -> None:
        return super().add(text)

    def extract(self, max_candidates=-1, min_cnt=10, max_length=5, min_score=1e-3, num_workers=1) -> List[NgramCandidate]:
        return super().extract(NgramCandidate, max_candidates, min_cnt, max_length, min_score, num_workers)

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
`orig`나 `error`는 완전한 음절 혹은 모음이나 자음을 포함할 수 있습니다. 자음의 경우 종성은 '\\\\'로 escape해주어야 합니다.

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
                ret.append('\\')
                ret.append(c)
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
continual_typo_cost: float
    .. versionadded:: 0.17.1

    연철에 대한 교정 비용. 기본값은 0으로, 연철에 대한 교정을 수행하지 않습니다.
lengthening_typo_cost: float
    .. versionadded:: 0.19.0

    장음화에 대한 교정 비용. 기본값은 0으로, 장음화에 대한 교정을 수행하지 않습니다.

Notes
-----
이 클래스의 인스턴스를 Kiwi 생성시의 typos 인자로 주면 Kiwi의 오타 교정 기능이 활성화됩니다.
```python
>>> from kiwipiepy import Kiwi, TypoTransformer, TypoDefinition
>>> typos = TypoTransformer([
    TypoDefinition(["ㅐ", "ㅔ"], ["ㅐ", "ㅔ"], 1.), # ㅐ 혹은 ㅔ를 ㅐ 혹은 ㅔ로 교체하여 오타를 생성. 생성 비용은 1
    TypoDefinition(["ㅔ"], ["ㅖ"], 2.), # ㅔ를 ㅖ로 교체하여 오타를 생성. 생성 비용은 2
])
>>> typos.generate('과제', 1.) # 생성 비용이 1.0이하인 오타들을 생성
[('과제', 0.0), ('과재', 1.0)]
>>> typos.generate('과제', 2.) # 생성 비용이 2.0이하인 오타들을 생성
[('과제', 0.0), ('과재', 1.0), ('과졔', 2.0)]

>>> kiwi = Kiwi(typos=typos, typo_cost_threshold=2.) # typos에 정의된 오타들을 교정 후보로 삼는 Kiwi 생성.
>>> kiwi.tokenize('과재를 했다') 
[Token(form='과제', tag='NNG', start=0, len=2), 
 Token(form='를', tag='JKO', start=2, len=1), 
 Token(form='하', tag='VV', start=4, len=1), 
 Token(form='었', tag='EP', start=4, len=1), 
 Token(form='다', tag='EF', start=5, len=1)]
```

서로 다른 오타 생성기를 합치기 위해서 `|` 연산자를 사용할 수 있습니다.
```python
>>> typos1 = TypoTransformer([
    TypoDefinition(["ㅐ", "ㅔ"], ["ㅐ", "ㅔ"], 1.),
    ])
>>> typos2 = TypoTransformer([
    TypoDefinition(["ㅔ"], ["ㅖ"], 2.),
    ])
>>> typos = typos1 | typos2 # typos1과 typos2를 합친 오타 생성기 생성
```
    '''

    def __init__(self,
        defs: List[TypoDefinition],
        continual_typo_cost: float = 0,
        lengthening_typo_cost: float = 0,
    ):
        try:
            assert continual_typo_cost >= 0, "continual_typo_cost should be zero or positive."
            assert lengthening_typo_cost >= 0, "lengthening_typo_cost should be zero or positive."
        except AssertionError as e:
            raise ValueError(*e.args)
        
        super().__init__(
            ((list(map(_convert_consonant, d.orig)), list(map(_convert_consonant, d.error)), d.cost, d.condition) for d in defs),
            continual_typo_cost,
            lengthening_typo_cost,
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

    def copy(self) -> 'TypoTransformer':
        '''.. versionadded:: 0.19.0
        
        현재 오타 생성기의 복사본을 생성합니다.'''
        return super().copy(TypoTransformer)

    def update(self, other:'TypoTransformer'):
        '''.. versionadded:: 0.19.0
        
        다른 오타 생성기의 오타 정의를 현재 오타 생성기에 추가합니다.'''
        super().update(other)

    def scale_cost(self, scale:float):
        '''오타 생성 비용을 scale배만큼 조정합니다.'''
        super().scale_cost(scale)

    def __or__(self, other:'TypoTransformer'):
        new_inst = self.copy()
        new_inst.update(other)
        return new_inst

    def __ior__(self, other:'TypoTransformer'):
        self.update(other)
        return self
    
    def __mul__(self, scale:float):
        new_inst = self.copy()
        new_inst.scale_cost(scale)
        return new_inst
    
    def __imul__(self, scale:float):
        self.scale_cost(scale)
        return self

    @property
    def defs(self) -> List[Tuple[str, str, float, Optional[str]]]:
        '''현재 오타 생성기의 정의자 목록'''
        return self._defs
    
    @property
    def continual_typo_cost(self) -> float:
        '''연철에 대한 교정 비용'''
        return self._continual_typo_cost
    
    @property
    def lengthening_typo_cost(self) -> float:
        '''.. versionadded:: 0.19.0
        
        장음화에 대한 교정 비용'''
        return self._lengthening_typo_cost

    def __repr__(self):
        defs = self._defs
        if len(defs) < 5:
            defs_str = ", ".join(map(repr, defs))
        else:
            defs_str = ", ".join(map(repr, defs[:5])) + ", ... ({} more)".format(len(defs) - 5)
        return "TypoTransformer([{}], continual_typo_cost={!r}, lengthening_typo_cost={!r})".format(defs_str, self._continual_typo_cost, self._lengthening_typo_cost)

class HSDataset(_HSDataset):
    pass

class MorphemeSet(_MorphemeSet):
    '''.. versionadded:: 0.15.0

    형태소 집합을 정의합니다. 정의된 형태소 집합은 `Kiwi.analyze`, `Kiwi.tokenize`, `Kiwi.split_into_sents`에서 blocklist 인자로 사용될 수 있습니다.

Parameters
----------
kiwi: Kiwi
    형태소 집합을 정의할 Kiwi의 인스턴스입니다.
morphs: Iterable[Union[str, Tuple[str, POSTag]]]
    집합에 포함될 형태소의 목록입니다. 형태소는 단일 `str`이나 `tuple`로 표기될 수 있습니다.

Notes
-----
형태소는 다음과 같이 크게 3가지 방법으로 표현될 수 있습니다.

```python
morphset = MorphemeSet([
    '고마움' # 형태만을 사용해 표현. 형태가 '고마움'인 모든 형태소가 이 집합에 포함됨
    '고마움/NNG' # 형태와 품사 태그를 이용해 표현. 형태가 '고마움'인 일반 명사만 이 집합에 포함됨
    ('고마움', 'NNG') # tuple로 분리해서 표현하는 것도 가능
])
```
    '''
    def __init__(self, 
        kiwi, 
        morphs:Iterable[Union[str, Tuple[str, POSTag]]]
    ):
        if not isinstance(kiwi, Kiwi):
            raise ValueError("`kiwi` must be an instance of `Kiwi`.")
        super().__init__(kiwi)
        self.kiwi = kiwi
        self.set = set(map(self._normalize, morphs))
        self._updated = False
    
    def __repr__(self):
        return f"MorphemeSet(kiwi, {repr(self.set)})"

    def __len__(self):
        return len(self.set)

    def _normalize(self, tagged_form):
        if isinstance(tagged_form, str):
            form, *tag = tagged_form.split('/', 1)
            tag = tag[0] if len(tag) else ''
            return form, tag
        elif isinstance(tagged_form, tuple):
            if len(tagged_form) == 2: return tagged_form
        
        raise ValueError("Morpheme should has a `str` or `Tuple[str, str]` type.")
    
    def _update_self(self):
        if self._updated: return
        super()._update(self.set)
        self._updated = True

class Kiwi(_Kiwi):
    '''Kiwi 클래스는 실제 형태소 분석을 수행하는 kiwipiepy 모듈의 핵심 클래스입니다.
이 클래스는 지연 초기화(Lazy initialization)를 사용합니다. 즉 `Kiwi` 인스턴스를 생성할 때에는 최소한의 초기화만 수행하고, 
실제 분석을 수행하는 함수(`tokenize`, `analyze` 등)를 호출할 때에 초기화가 완료됩니다.
따라서 분석 함수를 최초로 호출할 때에는 초기화에 추가적인 시간이 소요될 수 있습니다.
만약 미리 초기화를 수행하고 싶다면 인스턴스가 생성된 후에 빈 문자열에 대해서 `Kiwi.tokenize()` 함수를 호출하는 것을 권장합니다.

Parameters
----------
num_workers: int
    내부적으로 멀티스레딩에 사용할 스레드 개수. 0으로 설정시 시스템 내 가용한 모든 코어 개수만큼 스레드가 생성됩니다.
    멀티스레딩은 extract 계열 함수에서 단어 후보를 탐색할 때와 analyze 함수에서만 사용됩니다.
model_path: str
    읽어들일 모델 파일의 경로. 모델 파일의 위치를 옮긴 경우 이 값을 지정해주어야 합니다.

integrate_allormoph: bool
    True일 경우 음운론적 이형태를 통합하여 출력합니다. /아/와 /어/나 /았/과 /었/ 같이 앞 모음의 양성/음성에 따라 형태가 바뀌는 어미들을 하나로 통합하여 출력합니다. 기본값은 True입니다.
load_default_dict: bool
    True일 경우 인스턴스 생성시 자동으로 기본 사전을 불러옵니다. 기본 사전은 위키백과와 나무위키에서 추출된 고유 명사 표제어들로 구성되어 있습니다. 기본값은 True입니다.
load_typo_dict: bool
    .. versionadded:: 0.14.0
    
    True일 경우 인스턴스 생성시 자동으로 내장 오타 사전을 불러옵니다. 오타 사전은 자주 틀리는 오타 일부와 인터넷에서 자주 쓰이는 변형된 종결 어미로 구성되어 있습니다. 기본값은 True입니다.
load_multi_dict: bool
    .. versionadded:: 0.17.0
    
    True일 경우 인스턴스 생성시 자동으로 내장 다어절 사전을 불러옵니다. 다어절 사전은 WikiData에 등재된 고유 명사들로 구성되어 있습니다. 기본값은 True입니다.
model_type: str
    .. versionadded:: 0.13.0

    형태소 분석에 사용할 언어 모델을 지정합니다. `'knlm'`, `'sbg'` 중 하나를 선택할 수 있습니다. 기본값은 `'knlm'`입니다. 각 모델에 대한 자세한 설명은 <a href='#_4'>여기</a>를 참조하세요.

typos: Union[str, TypoTransformer]
    .. versionupdated:: 0.19.0

    교정에 사용할 오타 정보입니다. 기본값은 `None`으로 이 경우 오타 교정을 사용하지 않습니다. `TypoTransformer` 인스턴스를 입력하거나 약어로 다음 문자열을 입력할 수 있습니다.

    * `'basic'`: 기본 오타 정보(`kiwipiepy.basic_typos`)
    * `'continual'`: 연철 오타 정보(`kiwipiepy.continual_typos`)
    * `'basic_with_continual'`: 기본 오타 정보와 연철 함께 사용(`kiwipiepy.basic_typos_with_continual`)
    * `'lengthening'`: 장음화 오타 정보(`kiwipiepy.lengthening_typos`)
    * `'basic_with_continual_and_lengthening'`: 기본 오타 정보, 연철, 장음화 함께 사용(`kiwipiepy.basic_typos_with_continual_and_lengthening`)

    이에 대한 자세한 내용은 `kiwipiepy.TypoTransformer` 및 <a href='#_5'>여기</a>를 참조하세요.
typo_cost_threshold: float
    .. versionadded:: 0.13.0

    오타 교정시 고려할 최대 오타 비용입니다. 이 비용을 넘어서는 오타에 대해서는 탐색하지 않습니다. 기본값은 2.5입니다.
    '''

    def __init__(self, 
        num_workers: Optional[int] = None,
        model_path: Optional[str] = None,
        integrate_allomorph: Optional[bool] = None,
        load_default_dict: Optional[bool] = None,
        load_typo_dict: Optional[bool] = None,
        load_multi_dict: Optional[bool] = None,
        model_type: str = 'knlm',
        typos: Optional[Union[str, TypoTransformer]] = None,
        typo_cost_threshold: float = 2.5,
    ) -> None:
        if num_workers is None:
            num_workers = 0
        
        if integrate_allomorph is None:
            integrate_allomorph = True
        if load_default_dict is None:
            load_default_dict = True
        if load_typo_dict is None:
            load_typo_dict = True
        if load_multi_dict is None:
            load_multi_dict = True

        if model_type not in ('knlm', 'sbg'):
            raise ValueError("`model_type` should be one of ('knlm', 'sbg'), but {}".format(model_type))
        
        import kiwipiepy
        if typos == 'basic': 
            rtypos = kiwipiepy.basic_typos
        elif typos == 'continual':
            rtypos = kiwipiepy.continual_typos
        elif typos == 'basic_with_continual':
            rtypos = kiwipiepy.basic_typos_with_continual
        elif typos == 'lengthening':
            rtypos = kiwipiepy.lengthening_typos
        elif typos == 'basic_with_continual_and_lengthening':
            rtypos = kiwipiepy.basic_typos_with_continual_and_lengthening
        elif typos is None or isinstance(typos, TypoTransformer):
            rtypos = typos
        else:
            raise ValueError("`typos` should be one of ('basic', 'continual', 'basic_with_continual', 'lengthening', 'basic_with_continual_and_lengthening', TypoTransformer), but {}".format(typos))

        super().__init__(
            num_workers,
            model_path,
            integrate_allomorph,
            load_default_dict,
            load_typo_dict,
            load_multi_dict,
            (model_type=='sbg'),
            rtypos,
            typo_cost_threshold,
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
        self._model_path = model_path
        self._load_default_dict = load_default_dict
        self._load_typo_dict = load_typo_dict
        self._typos = typos
        self._pretokenized_pats : List[Tuple['re.Pattern', str, Any]] = []
        self._user_values : Dict[int, Any] = {}
        self._template_cache : Dict[str, Template] = {}

    def __repr__(self):
        return (
            f"Kiwi(num_workers={self.num_workers!r}, " 
            f"model_path={self._model_path!r}, "
            f"integrate_allomorph={self.integrate_allomorph!r}, "
            f"load_default_dict={self._load_default_dict!r}, "
            f"load_typo_dict={self._load_typo_dict!r}, "
            f"model_type={self.model_type!r}, "
            f"typos={self._typos!r}, "
            f"typo_cost_threshold={self.typo_cost_threshold!r}"
            f")"
        )

    def add_user_word(self,
        word:str,
        tag:POSTag = 'NNP',
        score:float = 0.,
        orig_word:Optional[str] = None,
        user_value:Optional[Any] = None,
    ) -> bool:
        '''현재 모델에 사용자 정의 형태소를 추가합니다.

..versionchanged:: 0.17.0
    0.17.0버전부터 공백이 포함된 단어(복합 명사 등)를 추가할 수 있게 되었습니다.        
        
Parameters
----------
word: str
    추가할 형태소
tag: str
    추가할 형태소의 품사 태그
score: float
    추가할 형태소의 가중치 점수. 
    해당 형태에 부합하는 형태소 조합이 여러 개가 있는 경우, 이 점수가 높을 단어가 더 우선권을 가집니다.
orig_word: str
    .. versionadded:: 0.11.0

    추가할 형태소의 원본 형태소.
    추가할 형태소가 특정 형태소의 변이형인 경우 이 인자로 원본 형태소를 넘겨줄 수 있습니다. 없는 경우 생략할 수 있습니다.
    `orig_word`가 주어진 경우 현재 사전 내에 `orig_word`/`tag` 조합의 형태소가 반드시 존재해야 하며, 그렇지 않으면 `ValueError` 예외를 발생시킵니다.
user_value: Any
    .. versionadded:: 0.16.0
    
    추가할 형태소의 사용자 지정값. 이 값은 형태소 분석 결과를 담는 `Token` 클래스의 `Token.user_value`값에 반영됩니다.
    또한 만약 `{'tag':'SPECIAL'}`와 같이 dict형태로 'tag'인 key를 제공하는 경우, 형태소 분석 결과의 tag값이 SPECIAL로 덮어씌워져서 출력됩니다.
Returns
-------
inserted: bool
    사용자 정의 형태소가 정상적으로 삽입된 경우 True, 이미 동일한 형태소가 존재하여 삽입되지 않은 경우 False를 반환합니다.

Notes
-----
공백을 포함하는 단어의 경우 단어 시작과 끝의 공백은 제거되며, 중간에 1개 이상의 공백이 연속하는 경우 공백 문자 하나로 정규화됩니다.
줄바꿈 문자, 탭 문자 등도 공백 문자로 취급되어 정규화됩니다.
예를 들어 `복합 명사`, ` 복합 명사 `, `복합   명사`, `복합\\n명사`는 모두 `복합 명사`로 정규화되어 동일하게 처리됩니다.
```python
>>> kiwi.add_user_word('복합 명사', 'NNP')
True
>>> kiwi.add_user_word(' 복합 명사 ', 'NNP') # 동일한 단어가 이미 삽입되어 있으므로 False 반환
False
>>> kiwi.add_user_word('복합   명사', 'NNP')
False
>>> kiwi.add_user_word('복합\\n명사', 'NNP')
False
```
        '''
        mid, inserted = super().add_user_word(word, tag, score, orig_word)
        self._user_values[mid] = user_value
        return inserted
    
    def add_pre_analyzed_word(self,
        form:str,
        analyzed:Iterable[Union[Tuple[str, POSTag], Tuple[str, POSTag, int, int]]],
        score:float = 0.,
    ) -> bool:
        '''.. versionadded:: 0.11.0

현재 모델에 기분석 형태소열을 추가합니다.

Parameters
----------
form: str
    추가할 형태
analyzed: Iterable[Union[Tuple[str, POSTag], Tuple[str, POSTag, int, int]]]
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
        analyzed = list(analyzed)
        if all(len(a) == 2 for a in analyzed) and ''.join(a[0] for a in analyzed) == form:
            new_analyzed = []
            cursor = 0
            for f, t in analyzed:
                p = form.index(f, cursor)
                if p < 0: break
                new_analyzed.append((f, t, p, p + len(f)))
                cursor = p
            if len(new_analyzed) == len(analyzed):
                analyzed = new_analyzed
        return super().add_pre_analyzed_word(form, analyzed, score)
    
    def add_re_word(self,
        pattern:Union[str, 're.Pattern'],
        pretokenized:Union[Callable[['re.Match'], Union[PretokenizedToken, List[PretokenizedToken]]], POSTag, PretokenizedToken, List[PretokenizedToken]],
        user_value:Optional[Any] = None,
    ) -> None:
        '''.. versionadded:: 0.16.0

현재 모델에 정규표현식 기반의 패턴 매칭을 사용한 형태소 목록을 삽입합니다.

Parameters
----------
pattern: Union[str, re.Pattern]
    정규표현식 문자열 혹은 `re.compile`로 컴파일된 정규표현식 객체. 형태소 분석시 이 정규표현식에 일치하는 패턴을 발견하면 항상 `pretokenized`에 따라 분석합니다.
pretokenized: Union[Callable[['re.Match'], Union[PretokenizedToken, List[PretokenizedToken]]], POSTag, PretokenizedToken, List[PretokenizedToken]]
    정규표현식으로 지정된 패턴이 분석될 형태를 지정합니다.
    POSTag, `PretokenizedToken`, `PretokenizedToken`의 리스트 또는 `re.Match`를 입력받아 `PretokenizedToken`의 리스트를 반환하는 콜백 함수 중 하나로 지정할 수 있습니다.

    이 값을 POSTag로 지정한 경우, 전체 패턴은 단 하나의 형태소로 분석되며 그때의 품사태그는 POSTag로 지정한 값을 따릅니다.
    PretokenizedToken로 지정한 경우, 전체 패턴은 단 하나의 형태소로 분석되며, 그때의 형태/품사태그/시작위치/끝위치는 PretokenizedToken로 지정한 값을 따릅니다.
    PretokenizedToken의 리스트로 지정한 경우, 전체 패턴읜 리스트에서 제시한 형태소 개수에 분할되어 분석되며, 각각의 형태소 정보는 리스트에 포함된 PretokenizedToken 값들을 따릅니다.
    마지막으로 콜백 함수로 지정한 경우, `pattern.search`의 결과가 함수의 인자로 제공되며, 이 인자를 처리한뒤 콜백 함수에서 반환하는 PretokenizedToken 값을 따라 형태소 분석 결과가 생성됩니다.
user_value: Any
    추가할 형태소의 사용자 지정값. 이 값은 형태소 분석 결과를 담는 `Token` 클래스의 `Token.user_value`값에 반영됩니다.
    또한 만약 `{'tag':'SPECIAL'}`와 같이 dict형태로 'tag'인 key를 제공하는 경우, 형태소 분석 결과의 tag값이 SPECIAL로 덮어씌워져서 출력됩니다.

Notes
-----
이 메소드는 분석할 텍스트 내에 분할되면 안되는 텍스트 영역이 있거나, 이미 분석된 결과가 포함된 텍스트를 분석하는 데에 유용합니다.

참고로 이 메소드는 형태소 분석에 앞서 전처리 단계에서 패턴 매칭을 수행하므로, 이를 통해 지정한 규칙들은 형태소 분석 모델보다 먼저 우선권을 갖습니다.
따라서 이 규칙으로 지정한 조건을 만족하는 문자열 패턴은 항상 이 규칙에 기반하여 처리되므로 맥락에 따라 다른 처리를 해야하는 경우 이 메소드를 사용하는 것을 권장하지 않습니다.

```python
>>> kiwi = Kiwi()
>>> text = '<평만경(平滿景)>이 사람을 시켜 <침향(沈香)> 10냥쭝을 바쳤으므로'
# <>로 둘러싸인 패턴은 전체를 NNP 태그로 분석하도록 설정
>>> kiwi.add_re_word(r'<[^>]+>', 'NNP') 
>>> kiwi.tokenize(text)
[Token(form='<평만경(平滿景)>', tag='NNP', start=0, len=10),
 Token(form='이', tag='JKS', start=10, len=1), 
 Token(form='사람', tag='NNG', start=12, len=2), 
 Token(form='을', tag='JKO', start=14, len=1), 
 Token(form='시키', tag='VV', start=16, len=2), 
 Token(form='어', tag='EC', start=17, len=1), 
 Token(form='<침향(沈香)>', tag='NNP', start=19, len=8), 
 Token(form='10', tag='SN', start=28, len=2), 
 Token(form='냥', tag='NNB', start=30, len=1), 
 Token(form='쭝', tag='NNG', start=31, len=1), 
 Token(form='을', tag='JKO', start=32, len=1), 
 Token(form='바치', tag='VV', start=34, len=2), 
 Token(form='었', tag='EP', start=35, len=1), 
 Token(form='으므로', tag='EC', start=36, len=3)]

>>> kiwi.clear_re_words() # 패턴 제거
# callback 함수를 입력하여 세부적인 조절이 가능
# 추출되는 패턴의 첫번째 괄호그룹을 형태소의 형태로 사용하도록 하여
# 분석 결과에 <, >가 포함되지 않도록 한다
>>> kiwi.add_re_word(r'<([^>]+)>', lambda m:PretokenizedToken(m.group(1), 'NNP', m.span(1)[0] - m.span(0)[0], m.span(1)[1] - m.span(0)[0]))
>>> kiwi.tokenize(text)
[Token(form='평만경(平滿景)', tag='NNP', start=0, len=10),
 Token(form='이', tag='MM', start=10, len=1), 
 Token(form='사람', tag='NNG', start=12, len=2), 
 Token(form='을', tag='JKO', start=14, len=1), 
 Token(form='시키', tag='VV', start=16, len=2), 
 Token(form='어', tag='EC', start=17, len=1), 
 Token(form='침향(沈香)', tag='NNP', start=19, len=8), 
 Token(form='10', tag='SN', start=28, len=2), 
 Token(form='냥', tag='NNB', start=30, len=1), 
 Token(form='쭝', tag='NNG', start=31, len=1), 
 Token(form='을', tag='JKO', start=32, len=1), 
 Token(form='바치', tag='VV', start=34, len=2), 
 Token(form='었', tag='EP', start=35, len=1), 
 Token(form='으므로', tag='EC', start=36, len=3)]

# 숫자 + 단위를 하나의 형태로 분석하도록 하는데에도 용이
>>> kiwi.add_re_word(r'[0-9]+냥쭝', 'NNG')
>>> kiwi.tokenize(text)
[Token(form='평만경(平滿景)', tag='NNP', start=0, len=10), 
 Token(form='이', tag='MM', start=10, len=1), 
 Token(form='사람', tag='NNG', start=12, len=2), 
 Token(form='을', tag='JKO', start=14, len=1), 
 Token(form='시키', tag='VV', start=16, len=2), 
 Token(form='어', tag='EC', start=17, len=1), 
 Token(form='침향(沈香)', tag='NNP', start=19, len=8), 
 Token(form='10냥쭝', tag='NNG', start=28, len=4), 
 Token(form='을', tag='NNG', start=32, len=1), 
 Token(form='바치', tag='VV', start=34, len=2), 
 Token(form='었', tag='EP', start=35, len=1), 
 Token(form='으므로', tag='EC', start=36, len=3)]

# 코드 영역의 분석을 방지하는 용도로도 사용이 가능
>>> import re
>>> text = """마크다운 코드가 섞인 문자열
```python
import kiwipiepy\\n```
입니다."""
>>> pat = re.compile(r'^```python\\n.*?^```', flags=re.DOTALL | re.MULTILINE)
# user_value를 지정하여 해당 패턴의 태그를 CODE로 덮어쓰기
>>> kiwi.add_re_word(pat, 'USER0', {'tag':'CODE'})
>>> kiwi.tokenize(text)
[Token(form='마크다운', tag='NNP', start=0, len=4),
 Token(form='코드', tag='NNG', start=5, len=2), 
 Token(form='가', tag='JKS', start=7, len=1), 
 Token(form='섞이', tag='VV', start=9, len=2), 
 Token(form='ᆫ', tag='ETM', start=10, len=1), 
 Token(form='문자열', tag='NNP', start=12, len=3), 
 Token(form='```python\\nimport kiwipiepy\\n```', tag='CODE', start=16, len=30), 
 Token(form='이', tag='VCP', start=47, len=1), 
 Token(form='ᆸ니다', tag='EF', start=47, len=3)]
```
        '''
        if isinstance(pattern, str):
            pattern = re.compile(pattern)
            
        self._pretokenized_pats.append((pattern, pretokenized, user_value))

    def clear_re_words(self):
        '''.. versionadded:: 0.16.0

`add_re_word`로 추가했던 정규표현식 패턴 기반 처리 규칙을 모두 삭제합니다.
        '''
        self._pretokenized_pats.clear()

    def add_rule(self,
        tag:POSTag,
        replacer:Callable[[str], str],
        score:float = 0.,
        user_value:Optional[Any] = None,
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
user_value: Any
    .. versionadded:: 0.16.0
    
    추가할 형태소의 사용자 지정값. 이 값은 형태소 분석 결과를 담는 `Token` 클래스의 `Token.user_value`값에 반영됩니다.
    또한 만약 `{'tag':'SPECIAL'}`와 같이 dict형태로 'tag'인 key를 제공하는 경우, 형태소 분석 결과의 tag값이 SPECIAL로 덮어씌워져서 출력됩니다.
Returns
-------
inserted_forms: List[str]
    규칙에 의해 새로 생성된 형태소의 `list`를 반환합니다.
        '''
        ret = super().add_rule(tag, replacer, score)
        if not ret: return []
        mids, inserted_forms = zip(*ret)
        for mid in mids:
            self._user_values[mid] = user_value
        return inserted_forms
    
    def add_re_rule(self,
        tag:POSTag,
        pattern:Union[str, 're.Pattern'],
        repl:Union[str, Callable],
        score:float = 0.,
        user_value:Optional[Any] = None,
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
user_value: Any
    .. versionadded:: 0.16.0
    
    추가할 형태소의 사용자 지정값. 이 값은 형태소 분석 결과를 담는 `Token` 클래스의 `Token.user_value`값에 반영됩니다.
    또한 만약 `{'tag':'SPECIAL'}`와 같이 dict형태로 'tag'인 key를 제공하는 경우, 형태소 분석 결과의 tag값이 SPECIAL로 덮어씌워져서 출력됩니다.
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
        return self.add_rule(tag, lambda x:pattern.sub(repl, x), score, user_value)

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
        min_cnt:int = 10,
        max_word_len:int = 10,
        min_score:float = 0.25,
        pos_score:float = -3.,
        lm_filter:bool = True,
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
    
    def extract_add_words(self,
        texts,
        min_cnt:int = 10,
        max_word_len:int = 10,
        min_score:float = 0.25,
        pos_score:float = -3.,
        lm_filter:bool = True,
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
    
    def _make_pretokenized_spans(self, override_pretokenized, text:str):
        span_groups = []
        for pattern, s, user_value in self._pretokenized_pats:
            spans = []
            if callable(s):
                for m in pattern.finditer(text):
                    spans.append((*m.span(), s(m), user_value))
            else:
                for m in pattern.finditer(text):
                    spans.append((*m.span(), s, user_value))
            if spans:
                span_groups.append(spans)


        if callable(override_pretokenized):
            spans = override_pretokenized(text)
            if spans: 
                if not all(0 <= s <= e <= len(text) for s, e, *_ in spans):
                    raise ValueError("All spans must be valid range of text")
                span_groups.append(spans)
        elif override_pretokenized is not None:
            spans = override_pretokenized
            if spans: 
                if not all(0 <= s <= e <= len(text) for s, e, *_ in spans):
                    raise ValueError("All spans must be valid range of text")
                span_groups.append(spans)
        
        return span_groups

    def analyze(self,
        text:Union[str, Iterable[str]],
        top_n:int = 1,
        match_options:int = Match.ALL,
        normalize_coda:bool = False,
        z_coda:bool = True,
        split_complex:bool = False,
        compatible_jamo:bool = False,
        blocklist:Optional[Union[MorphemeSet, Iterable[str]]] = None,
        pretokenized:Optional[Union[Callable[[str], PretokenizedTokenList], PretokenizedTokenList]] = None,
    ) -> List[Tuple[List[Token], float]]:
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
    이 인자는 `Kiwi.tokenize`에서와 동일한 역할을 수행합니다.
normalize_coda: bool
    이 인자는 `Kiwi.tokenize`에서와 동일한 역할을 수행합니다.
z_coda: bool
    이 인자는 `Kiwi.tokenize`에서와 동일한 역할을 수행합니다.
split_complex: bool
    이 인자는 `Kiwi.tokenize`에서와 동일한 역할을 수행합니다.
compatible_jamo: bool
    이 인자는 `Kiwi.tokenize`에서와 동일한 역할을 수행합니다.
blocklist: Union[Iterable[str], MorphemeSet]
    이 인자는 `Kiwi.tokenize`에서와 동일한 역할을 수행합니다.
pretokenized: Union[Callable[[str], PretokenizedTokenList], PretokenizedTokenList]
    이 인자는 `Kiwi.tokenize`에서와 동일한 역할을 수행합니다.

Returns
-------
result: List[Tuple[List[Token], float]]
    text를 str으로 준 경우.
    분석 결과는 최대 `top_n`개 길이의 리스트로 반환됩니다. 리스트의 각 항목은 `(분석 결과, 분석 점수)`로 구성된 튜플입니다. 
    `분석 결과`는 `Token`의 리스트로 구성됩니다.

results: Iterable[List[Tuple[List[Token], float]]]
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
        if z_coda:
            match_options |= Match.Z_CODA
        if split_complex:
            match_options |= Match.SPLIT_COMPLEX
        if compatible_jamo:
            match_options |= Match.COMPATIBLE_JAMO
        if isinstance(blocklist, MorphemeSet):
            if blocklist.kiwi != self: 
                warnings.warn("This `MorphemeSet` isn't based on current Kiwi object.")
                blocklist = MorphemeSet(self, blocklist.set)
        elif blocklist is not None:
            blocklist = MorphemeSet(self, blocklist)
        
        if blocklist: blocklist._update_self()

        if not isinstance(text, str) and pretokenized and not callable(pretokenized):
            raise ValueError("`pretokenized` must be a callable if `text` is an iterable of str.")
        pretokenized = partial(self._make_pretokenized_spans, pretokenized) if self._pretokenized_pats or pretokenized else None

        return super().analyze(text, top_n, match_options, False, blocklist, pretokenized)
    
    def morpheme(self,
        idx:int,
    ):
        return super().morpheme(idx)
    
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

    @property
    def typo_cost_threshold(self):
        '''.. versionadded:: 0.15.0

오타 교정시 고려할 최대 오타 비용입니다. 이 비용을 넘어서는 오타에 대해서는 탐색하지 않습니다. 기본값은 2.5입니다.
        '''
        return self._typo_cost_threshold
    
    @typo_cost_threshold.setter
    def typo_cost_threshold(self, v:float):
        if v <= 0: raise ValueError("`typo_cost_threshold` must greater than 0")
        self._typo_cost_threshold = float(v)

    def _tokenize(self, 
        text:Union[str, Iterable[str]], 
        match_options:int = Match.ALL,
        normalize_coda:bool = False,
        z_coda:bool = True,
        split_complex:bool = False,
        compatible_jamo:bool = False,
        split_sents:bool = False,
        stopwords:Optional[Stopwords] = None,
        echo:bool = False,
        blocklist:Optional[Union[Iterable[str], MorphemeSet]] = None,
        pretokenized:Optional[Union[Callable[[str], PretokenizedTokenList], PretokenizedTokenList]] = None,
    ):
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
        if z_coda:
            match_options |= Match.Z_CODA
        if split_complex:
            match_options |= Match.SPLIT_COMPLEX
        if compatible_jamo:
            match_options |= Match.COMPATIBLE_JAMO

        if isinstance(blocklist, MorphemeSet):
            if blocklist.kiwi != self: 
                warnings.warn("This `MorphemeSet` isn't based on current Kiwi object.")
                blocklist = MorphemeSet(self, blocklist.set)
        elif blocklist is not None:
            blocklist = MorphemeSet(self, blocklist)
        
        if blocklist: blocklist._update_self()

        if not isinstance(text, str) and pretokenized and not callable(pretokenized):
            raise ValueError("`pretokenized` must be a callable if `text` is an iterable of str.")

        pretokenized = partial(self._make_pretokenized_spans, pretokenized) if self._pretokenized_pats or pretokenized else None

        if isinstance(text, str):
            echo = False
            return _refine_result(super().analyze(text, 1, match_options, False, blocklist, pretokenized))
        
        return map(_refine_result_with_echo if echo else _refine_result, super().analyze(text, 1, match_options, echo, blocklist, pretokenized))

    def tokenize(self, 
        text:Union[str, Iterable[str]], 
        match_options:int = Match.ALL,
        normalize_coda:bool = False,
        z_coda:bool = True,
        split_complex:bool = False,
        compatible_jamo:bool = False,
        split_sents:bool = False,
        stopwords:Optional[Stopwords] = None,
        echo:bool = False,
        blocklist:Optional[Union[Iterable[str], MorphemeSet]] = None,
        pretokenized:Optional[Union[Callable[[str], PretokenizedTokenList], PretokenizedTokenList]] = None,
    ) -> Union[List[Token], Iterable[List[Token]], List[List[Token]], Iterable[List[List[Token]]]]:
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
z_coda: bool

    .. versionadded:: 0.15.0

    기본값은 True로, True인 경우 '먹었어욥' => `먹었어요 + ㅂ`, '우리집에성' => `우리집에서 + ㅇ`과 같이 
    조사 및 어미에 덧붙는 받침을 'z_coda'라는 태그로 분리합니다. 
    False로 설정 시 덧붙는 받침 분리를 수행하지 않는 대신 분석 속도가 향상됩니다.
split_complex: bool

    .. versionadded:: 0.15.0

    True인 경우 둘 이상의 형태소로 더 잘게 분할될 수 있는 경우 형태소를 최대한 분할합니다. 
    예를 들어 '고마움을 전하다'의 경우 split_complex=False인 경우 `고마움/NNG 을/JKO 전하/VV 다/EF`와 같이 분석되지만,
    split_complex=True인 경우 `고맙/VA-I 음/ETN 을/JKO 전하/VV 다/EF`처럼 `고마움/NNG`이 더 잘게 분석됩니다.
compatible_jamo: bool

    .. versionadded:: 0.18.1

    True인 경우 분석 결과의 첫가끝 자모를 호환용 자모로 변환하여 출력합니다.
    예를 들어 "ᆫ다/EF"는 "ㄴ다/EF"로, "ᆯ/ETM"은 "ㄹ/ETM"으로 변환됩니다.
split_sents: bool
    .. versionadded:: 0.10.3

    True인 경우 형태소 리스트를 문장별로 묶어서 반환합니다. 자세한 내용은 아래의 Returns 항목을 참조하세요.
stopwords: Stopwords
    .. versionadded:: 0.10.3

    이 인자로 None이 아닌 `kiwipiepy.utils.Stopwords` 객체를 줄 경우, 형태소 분석 결과 중 그 객체에 포함되는 불용어를 제외한 나머지 결과만을 반환합니다.
echo: bool
    .. versionadded:: 0.11.2

    이 값이 True이고 `text`를 str의 Iterable로 준 경우, 분석 결과뿐만 아니라 원본 입력을 함께 반환합니다. `text`가 단일 str인 경우 이 인자는 무시됩니다.
blocklist: Union[MorphemeSet, Iterable[str]]

    .. versionadded:: 0.15.0

    분석 시 후보로 나타나는 걸 금지할 형태소 목록을 지정합니다. 
    예를 들어, blocklist=['고마움']으로 지정하는 경우, 분석 결과에서 '고마움'이라는 형태소가 등장하는 것을 막아
    split_complex의 예시에서처럼 '고마움을 전하다'가 `고맙/VA-I 음/ETN 을/JKO 전하/VV 다/EF`처럼 분석되도록 강요할 수 있습니다.
    blocklist는 `MorphemeSet` 혹은 `set`으로 지정할 수 있으며, 
    동일한 blocklist가 반복적으로 사용되는 경우 사전에 `MorphemeSet`을 만들고 이를 재사용하는게 효율적입니다.
pretokenized: Union[Callable[[str], PretokenizedTokenList], PretokenizedTokenList]

    .. versionadded:: 0.16.0

    형태소 분석에 앞서 텍스트 내 특정 구간의 형태소 분석 결과를 미리 정의합니다. 이 값에 의해 정의된 텍스트 구간은 항상 해당 방법으로만 토큰화됩니다.
    이 값은 str을 입력 받아 `PretokenizedTokenList`를 반환하는 `Callable`로 주어지거나, `PretokenizedTokenList` 값 단독으로 주어질 수 있습니다.
    `text`가 `Iterable[str]`인 경우 `pretokenized`는 None 혹은 `Callable`로 주어져야 합니다. 자세한 것은 아래 Notes의 예시를 참조하십시오.
Returns
-------
result: List[Token]
    `split_sents=False`일때 text를 str으로 준 경우.
    `Token`의 리스트를 반환합니다.

results: Iterable[List[Token]]
    `split_sents=False`일때 text를 Iterable[str]으로 준 경우.
    반환값은 `result`의 iterator로 주어집니다. iterator가 차례로 반환하는 분석결과 값은 입력으로 준 text의 순서와 동일합니다.

results_with_echo: Iterable[Tuple[List[Token], str]]
    `split_sents=False`이고 `echo=True`일때 text를 Iterable[str]으로 준 경우.
    반환값은 (`result`의 iterator, `raw_input`)으로 주어집니다. iterator가 차례로 반환하는 분석결과 값은 입력으로 준 text의 순서와 동일합니다.

result_by_sent: List[List[Token]]
    `split_sents=True`일때 text를 str으로 준 경우.
    형태소 분석 결과가 문장별로 묶여서 반환됩니다.
    즉, 전체 문장이 n개라고 할 때, `result_by_sent[0] ~ result_by_sent[n-1]`에는 각 문장별 분석 결과가 들어갑니다.

results_by_sent: Iterable[List[List[Token]]]
    `split_sents=True`일때 text를 Iterable[str]으로 준 경우.
    반환값은 `result_by_sent`의 iterator로 주어집니다. iterator가 차례로 반환하는 분석결과 값은 입력으로 준 text의 순서와 동일합니다.

results_by_sent_with_echo: Iterable[Tuple[List[List[Token]], str]]
    `split_sents=True`이고 `echo=True`일때 text를 Iterable[str]으로 준 경우.
    반환값은 (`result_by_sent`의 iterator, `raw_input`)으로 주어집니다. iterator가 차례로 반환하는 분석결과 값은 입력으로 준 text의 순서와 동일합니다.

Notes
-----

```python
>>> kiwi.tokenize("안녕하세요 형태소 분석기 키위입니다.")
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
>>> kiwi.tokenize("ㅋㅋㅋ 이런 것도 분석이 될까욬ㅋㅋ?", normalize_coda=True)
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
>>> from kiwipiepy.utils import Stopwords
>>> stopwords = Stopwords()
>>> kiwi.tokenize("분석 결과에서 불용어만 제외하고 출력할 수도 있다.", stopwords=stopwords)
[Token(form='분석', tag='NNG', start=0, len=2),
 Token(form='결과', tag='NNG', start=3, len=2),
 Token(form='불', tag='XPN', start=8, len=1),
 Token(form='용어', tag='NNG', start=9, len=2),
 Token(form='제외', tag='NNG', start=13, len=2),
 Token(form='출력', tag='NNG', start=18, len=2)]

# pretokenized 값을 지정해 특정 구간의 분석 결과를 직접 설정할 수 있습니다.
>>> text = "드디어패트와 매트가 2017년에 국내 개봉했다."
>>> kiwi.tokenize(text, pretokenized=[
        (3, 9), # 시작지점과 끝지점만 지정
        (11, 16, 'NNG'), #  시작지점과 끝지점, 품사 태그를 지정
    ])
[Token(form='드디어', tag='MAG', start=0, len=3), 
 Token(form='패트와 매트', tag='NNP', start=3, len=6), 
 Token(form='가', tag='JKS', start=9, len=1), 
 Token(form='2017년', tag='NNG', start=11, len=5), 
 Token(form='에', tag='JKB', start=16, len=1), 
 Token(form='국내', tag='NNG', start=18, len=2), 
 Token(form='개봉', tag='NNG', start=21, len=2), 
 Token(form='하', tag='XSV', start=23, len=1), 
 Token(form='었', tag='EP', start=23, len=1), 
 Token(form='다', tag='EF', start=24, len=1), 
 Token(form='.', tag='SF', start=25, len=1)]
# 시작지점과 끝지점만 지정한 경우 해당 구간은 한 덩어리로 묶여서 분석되며, 
#  그때의 품사태그는 모델이 알아서 선택합니다.
# 시작지점, 끝지점에 품사 태그까지 지정한 경우, 해당 구간은 반드시 그 품사태그로 분석됩니다.

# 각 구간의 분석 결과를 PretokenizedToken를 이용해 더 구체적으로 명시하는 것도 가능합니다.
>>> res = kiwi.tokenize(text, pretokenized=[
        (3, 5, PretokenizedToken('페트', 'NNB', 0, 2)),
        (21, 24, [PretokenizedToken('개봉하', 'VV', 0, 3), PretokenizedToken('었', 'EP', 2, 3)])
    ])
[Token(form='드디어', tag='MAG', start=0, len=3), 
 Token(form='페트', tag='NNB', start=3, len=2), 
 Token(form='와', tag='JC', start=5, len=1), 
 Token(form='매트', tag='NNG', start=7, len=2), 
 Token(form='가', tag='JKS', start=9, len=1), 
 Token(form='2017', tag='SN', start=11, len=4), 
 Token(form='년', tag='NNB', start=15, len=1), 
 Token(form='에', tag='JKB', start=16, len=1), 
 Token(form='국내', tag='NNG', start=18, len=2), 
 Token(form='개봉하', tag='VV', start=21, len=3), 
 Token(form='었', tag='EP', start=23, len=1), 
 Token(form='다', tag='EF', start=24, len=1), 
 Token(form='.', tag='SF', start=25, len=1)]
```
        '''
        return self._tokenize(text, match_options, normalize_coda, z_coda, split_complex, compatible_jamo, 
                              split_sents, stopwords, echo, 
                              blocklist=blocklist, 
                              pretokenized=pretokenized
        )

    def split_into_sents(self, 
        text:Union[str, Iterable[str]], 
        match_options:int = Match.ALL, 
        normalize_coda:bool = False,
        z_coda:bool = True,
        split_complex:bool = False,
        compatible_jamo:bool = False,
        stopwords:Optional[Stopwords] = None,
        blocklist:Optional[Union[Iterable[str], MorphemeSet]] = None,
        return_tokens:bool = False,
        return_sub_sents:bool = True,
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
    이 인자는 `Kiwi.tokenize`에서와 동일한 역할을 수행합니다.
normalize_coda: bool
    이 인자는 `Kiwi.tokenize`에서와 동일한 역할을 수행합니다.
z_coda: bool
    이 인자는 `Kiwi.tokenize`에서와 동일한 역할을 수행합니다.
split_complex: bool
    이 인자는 `Kiwi.tokenize`에서와 동일한 역할을 수행합니다.
compatible_jamo: bool
    이 인자는 `Kiwi.tokenize`에서와 동일한 역할을 수행합니다.
stopwords: Stopwords

    .. versionadded:: 0.16.0

    이 인자는 `Kiwi.tokenize`에서와 동일한 역할을 수행합니다.
blocklist: Union[Iterable[str], MorphemeSet]
    이 인자는 `Kiwi.tokenize`에서와 동일한 역할을 수행합니다.
return_tokens: bool
    True인 경우 문장별 형태소 분석 결과도 함께 반환합니다.
return_sub_sents: bool
    
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
>>> kiwi.split_into_sents("여러 문장으로 구성된 텍스트네 이걸 분리해줘")
[Sentence(text='여러 문장으로 구성된 텍스트네', start=0, end=16, tokens=None, subs=[]),
 Sentence(text='이걸 분리해줘', start=17, end=24, tokens=None, subs=[])]

>>> kiwi.split_into_sents("여러 문장으로 구성된 텍스트네 이걸 분리해줘", return_tokens=True)
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
>>> kiwi.split_into_sents("회사의 정보 서비스를 책임지고 있는 로웬버그John Loewenberg는" 
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
        def _filter_tokens(tokens):
            return tokens if stopwords is None else stopwords.filter(tokens)

        def _make_result(arg):
            sents, raw_input = arg
            ret = []
            for sent in sents:
                start = sent[0].start
                end = sent[-1].end
                tokens = _filter_tokens(sent) if return_tokens else None
                subs = None
                if return_sub_sents:
                    subs = []
                    sub_toks = []
                    last = 0
                    for tok in sent:
                        if tok.sub_sent_position != last:
                            if last:
                                subs.append(Sentence(raw_input[sub_start:last_end], sub_start, last_end, _filter_tokens(sub_toks) if return_tokens else None, None))
                                sub_toks = []
                            sub_start = tok.start
                        if tok.sub_sent_position:
                            sub_toks.append(tok)
                        last = tok.sub_sent_position
                        last_end = tok.end
                ret.append(Sentence(raw_input[start:end], start, end, tokens, subs))
            return ret

        if isinstance(text, str):
            return _make_result((self._tokenize(text, 
                                                match_options=match_options, 
                                                normalize_coda=normalize_coda, 
                                                z_coda=z_coda, 
                                                split_complex=split_complex, 
                                                compatible_jamo=compatible_jamo,
                                                blocklist=blocklist, 
                                                split_sents=True), text))

        return map(_make_result, self._tokenize(text, 
                                                match_options=match_options, 
                                                normalize_coda=normalize_coda, 
                                                z_coda=z_coda, 
                                                split_complex=split_complex, 
                                                compatible_jamo=compatible_jamo,
                                                blocklist=blocklist, 
                                                split_sents=True, 
                                                echo=True))

    def glue(self,
        text_chunks:Iterable[str],
        insert_new_lines:Optional[Iterable[bool]] = None,
        return_space_insertions:bool = False,
    ) -> Union[str, Tuple[str, List[bool]]]:
        '''..versionadded:: 0.11.1

여러 텍스트 조각을 하나로 합치되, 문맥을 고려해 적절한 공백을 사이에 삽입합니다.

Parameters
----------
text_chunks: Iterable[str]
    합칠 텍스트 조각들의 목록입니다.
insert_new_lines: Iterable[bool]

    ..versionadded:: 0.15.0

    합칠 때 공백 대신 줄바꿈을 사용할지 여부를 각 텍스트 조각별로 설정합니다. `insert_new_lines`의 길이는 `text_chunks`의 길이와 동일해야 합니다.
    생략 시 줄바꿈을 사용하지 않고 공백만을 사용합니다.
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
>>> kiwi.glue([
    "그러나  알고보니 그 봉",
    "지 안에 있던 것은 바로",
    "레몬이었던 것이다."])
"그러나  알고보니 그 봉지 안에 있던 것은 바로 레몬이었던 것이다."

>>> kiwi.glue([
    "그러나  알고보니 그 봉",
    "지 안에 있던 것은 바로",
    "레몬이었던 것이다."], return_space_insertions=True)
("그러나  알고보니 그 봉지 안에 있던 것은 바로 레몬이었던 것이다.", [False, True])
```
        '''

        all_chunks = []
        def _zip_consequences(it):
            try:
                prev = next(it).strip()
            except StopIteration:
                return

            all_chunks.append(prev)
            for s in it:
                s = s.strip()
                yield prev + ' ' + s
                yield prev + s
                prev = s
                all_chunks.append(prev)
        
        def _repeat_false():
            while 1:
                yield False

        riter = super().analyze(_zip_consequences(iter(text_chunks)), 1, Match.ALL, False, None, None)
            
        if insert_new_lines is None: 
            insert_new_lines = _repeat_false()
        else:
            insert_new_lines = iter(insert_new_lines)
        i = 0
        ret = []
        space_insertions = []
        try:
            while 1:
                _, score_with_space = next(riter)[0]
                _, score_without_space = next(riter)[0]
                is_new_line = next(insert_new_lines)
                ret.append(all_chunks[i])
                if score_with_space >= score_without_space or re.search(r'[0-9A-Za-z]$', all_chunks[i]):
                    ret.append('\n' if is_new_line else ' ')
                    space_insertions.append(True)
                else:
                    space_insertions.append(False)
                i += 1
        except StopIteration:
            if i < len(all_chunks):
                ret.append(all_chunks[i])
        
        if return_space_insertions:
            return ''.join(ret), space_insertions
        else:
            return ''.join(ret)

    def space(self,
        text:Union[str, Iterable[str]],
        reset_whitespace:bool = False,
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
>>> kiwi.space("띄어쓰기없이작성된텍스트네이걸교정해줘")
"띄어쓰기 없이 작성된 텍스트네 이걸 교정해 줘."
>>> kiwi.space("띄 어 쓰 기 문 제 가 있 습 니 다")
"띄어 쓰기 문 제 가 있 습 니 다"
>>> kiwi.space_tolerance = 2 # 형태소 내 공백을 최대 2개까지 허용
>>> kiwi.space("띄 어 쓰 기 문 제 가 있 습 니 다")
"띄어 쓰기 문제가 있습니다"
>>> kiwi.space("띄 어 쓰 기 문 제 가 있 습 니 다", reset_whitespace=True) # 기존 공백 전부 무시
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
            return _space((super().analyze(text, 1, Match.ALL | Match.Z_CODA, False, None, None), text))
        else:
            if reset_whitespace: text = map(_reset, text)
            return map(_space, super().analyze(text, 1, Match.ALL | Match.Z_CODA, True, None, None))

    def join(self, 
        morphs:Iterable[Tuple[str, str]],
        lm_search:bool = True,
        return_positions:bool = False,
    ) -> str:
        '''..versionadded:: 0.12.0

형태소들을 결합하여 문장으로 복원합니다. 
조사나 어미는 앞 형태소에 맞춰 적절한 형태로 변경됩니다.

Parameters
----------
morphs: Iterable[Union[Token, Tuple[str, str], Tuple[str, str, bool]]]
    결합할 형태소의 목록입니다. 
    각 형태소는 `Kiwi.tokenize`에서 얻어진 `Token` 타입이거나, 
    (형태, 품사) 혹은 (형태, 품사, 왼쪽 띄어쓰기 유무)로 구성된 `tuple` 타입이어야 합니다.
lm_search: bool
    둘 이상의 형태로 복원 가능한 모호한 형태소가 있는 경우, 이 값이 True면 언어 모델 탐색을 통해 최적의 형태소를 선택합니다.
    False일 경우 탐색을 실시하지 않지만 더 빠른 속도로 복원이 가능합니다.
return_positions: bool
    ..versionadded:: 0.17.0

    True인 경우, 각 형태소의 시작 위치와 끝 위치를 `List[Tuple[int, int]]`로 반환합니다.
    
Returns
-------
result: str
    입력 형태소의 결합 결과를 반환합니다.
positions: List[Tuple[int, int]]
    이 값은 `return_positions=True`인 경우에만 반환됩니다.
    결합된 문자열 상에서 각 형태소의 시작 위치와 끝 위치를 알려줍니다.
    
Notes
-----
`Kiwi.join`은 형태소를 결합할 때 `Kiwi.space`에서 사용하는 것과 유사한 규칙을 사용하여 공백을 적절히 삽입합니다.
형태소 그 자체에는 공백 관련 정보가 포함되지 않으므로
특정 텍스트를 `Kiwi.tokenize`로 분석 후 다시 `Kiwi.join`으로 결합하여도 원본 텍스트가 그대로 복원되지는 않습니다.


```python
>>> kiwi.join([('덥', 'VA'), ('어', 'EC')])
'더워'
>>> tokens = kiwi.tokenize("분석된결과를 다시합칠수있다!")
# 형태소 분석 결과를 복원. 
# 복원 시 공백은 규칙에 의해 삽입되므로 원문 텍스트가 그대로 복원되지는 않음.
>>> kiwi.join(tokens)
'분석된 결과를 다시 합칠 수 있다!'
>>> tokens[3]
Token(form='결과', tag='NNG', start=4, len=2)
>>> tokens[3] = ('내용', 'NNG') # 4번째 형태소를 결과->내용으로 교체
>>> kiwi.join(tokens) # 다시 join하면 결과를->내용을 로 교체된 걸 확인 가능
'분석된 내용을 다시 합칠 수 있다!'

# 불규칙 활용여부가 모호한 경우 lm_search=True인 경우 맥락을 고려해 최적의 후보를 선택합니다.
>>> kiwi.join([('길', 'NNG'), ('을', 'JKO'), ('묻', 'VV'), ('어요', 'EF')])
'길을 물어요'
>>> kiwi.join([('흙', 'NNG'), ('이', 'JKS'), ('묻', 'VV'), ('어요', 'EF')])
'흙이 묻어요'
# lm_search=False이면 탐색을 실시하지 않습니다.
>>> kiwi.join([('길', 'NNG'), ('을', 'JKO'), ('묻', 'VV'), ('어요', 'EF')], lm_search=False)
'길을 묻어요'
>>> kiwi.join([('흙', 'NNG'), ('이', 'JKS'), ('묻', 'VV'), ('어요', 'EF')], lm_search=False)
'흙이 묻어요'
# 동사/형용사 품사 태그 뒤에 -R(규칙 활용), -I(불규칙 활용)을 덧붙여 활용법을 직접 명시할 수 있습니다.
>>> kiwi.join([('묻', 'VV-R'), ('어요', 'EF')])
'묻어요'
>>> kiwi.join([('묻', 'VV-I'), ('어요', 'EF')])
'물어요'

# 0.15.2버전부터는 Tuple의 세번째 요소로 띄어쓰기 유무를 지정할 수 있습니다. 
# True일 경우 강제로 띄어쓰기, False일 경우 강제로 붙여쓰기를 수행합니다.
>>> kiwi.join([('길', 'NNG'), ('을', 'JKO', True), ('묻', 'VV'), ('어요', 'EF')])
'길 을 물어요'
>>> kiwi.join([('길', 'NNG'), ('을', 'JKO'), ('묻', 'VV', False), ('어요', 'EF')])
'길을물어요'

# 과거형 선어말어미를 제거하는 예시
>>> remove_past = lambda s: kiwi.join(t for t in kiwi.tokenize(s) if t.tagged_form != '었/EP')
>>> remove_past('먹었다')
'먹다'
>>> remove_past('먼 길을 걸었다')
'먼 길을 걷다'
>>> remove_past('전화를 걸었다.')
'전화를 걸다.'
```
        '''
        return super().join(morphs, lm_search, return_positions)

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

    def template(self,
        format_str:str,
        cache:bool = True,
    ) -> Template:
        '''..versionadded:: 0.16.1

한국어 형태소를 고려한 문자열 템플릿을 생성합니다. 
이를 사용하면 Python의 `str.format`과 거의 동일한 문법을 사용하여 빈 칸에 문자열을 채우는 것이 가능합니다.

Parameters
----------
format_str: str
    템플릿 문자열입니다. 이 문자열은 치환 필드를 {}로 나타냅니다. 템플릿 문자열의 구체적인 문법에 대해서는
    https://docs.python.org/ko/3/library/string.html#formatstrings 를 참고하세요.
cache: bool
    True인 경우 같은 포맷 문자열에 대해 이 메소드가 반환하는 템플릿 객체를 보관해둡니다.
    이 경우 동일한 템플릿 객체를 여러 번 생성할 때 더 빠른 속도로 생성이 가능해집니다. 기본값은 True입니다.

Returns
-------
template: kiwipiepy.Template
    템플릿 객체를 반환합니다.

Notes
-----
이 메소드는 한국어로 구성된 템플릿의 빈 칸을 채우는 데에 유용하게 사용될 수 있습니다.
특히 빈 칸 뒤에 조사나 어미가 오는 경우, 이 메소드를 사용하면 조사나 어미가 앞 형태소에 맞춰 적절히 조정됩니다.

```python
>>> kiwi = Kiwi()
# 빈칸은 {}로 표시합니다. 
# 이 자리에 형태소 혹은 기타 Python 객체가 들어가서 문자열을 완성시키게 됩니다.
>>> tpl = kiwi.template("{}가 {}을 {}었다.")

# template 객체는 format 메소드를 제공합니다. 
# 이 메소드를 통해 빈 칸을 채울 수 있습니다.
# 형태소는 `kiwipiepy.Token` 타입이거나 
# (형태, 품사) 혹은 (형태, 품사, 왼쪽 띄어쓰기 유무)로 구성된 tuple 타입이어야 합니다.
>>> tpl.format(("나", "NP"), ("공부", "NNG"), ("하", "VV"))
'내가 공부를 했다.'

>>> tpl.format(("너", "NP"), ("밥", "NNG"), ("먹", "VV"))
'네가 밥을 먹었다.'

>>> tpl.format(("우리", "NP"), ("길", "NNG"), ("묻", "VV-I"))
'우리가 길을 물었다.'

# 형태소가 아닌 Python 객체가 입력되는 경우 `str.format`과 동일하게 동작합니다.
>>> tpl.format(5, "str", {"dict":"dict"})
"5가 str를 {'dict': 'dict'}었다."

# 입력한 객체가 형태소가 아닌 Python 객체로 처리되길 원하는 경우 !s 변환 플래그를 사용합니다.
>>> tpl = kiwi.template("{!s}가 {}을 {}었다.")
>>> tpl.format(("나", "NP"), ("공부", "NNG"), ("하", "VV"))
"('나', 'NP')가 공부를 했다."

# Python 객체에 대해서는 `str.format`과 동일한 서식 지정자를 사용할 수 있습니다.
>>> tpl = kiwi.template("{:.5f}가 {!r}을 {}었다.")
>>> tpl.format(5, "str", {"dict":"dict"})
"5.00000가 'str'를 {'dict': 'dict'}었다."

# 서식 지정자가 주어진 칸에 형태소를 대입할 경우 ValueError가 발생합니다.
>>> tpl.format(("우리", "NP"), "str", ("묻", "VV-I"))
ValueError: cannot specify format specifier for Kiwi Token

# 치환 필드에 index나 name을 지정하여 대입 순서를 설정할 수 있습니다.
>>> tpl = kiwi.template("{0}가 {obj}를 {verb}\ㄴ다. {1}는 {obj}를 안 {verb}었다.")
>>> tpl.format(
    [("우리", "NP"), ("들", "XSN")], 
    [("너희", "NP"), ("들", "XSN")], 
    obj=("길", "NNG"), 
    verb=("묻", "VV-I")
)
'우리들이 길을 묻는다. 너희들은 길을 안 물었다.'

# 위의 예시처럼 종성 자음은 호환용 자모 코드 앞에 \\로 이스케이프를 사용해야합니다.
# 그렇지 않으면 종성이 아닌 초성으로 인식됩니다.
>>> tpl = kiwi.template("{0}가 {obj}를 {verb}ㄴ다. {1}는 {obj}를 안 {verb}었다.")
>>> tpl.format(
    [("우리", "NP"), ("들", "XSN")], 
    [("너희", "NP"), ("들", "XSN")], 
    obj=("길", "NNG"), 
    verb=("묻", "VV-I")
)
'우리들이 길을 묻 ᄂ이다. 너희들은 길을 안 물었다.'
```

        '''
        if not cache:
            return Template(self, format_str)
        
        try:
            return self._template_cache[format_str]
        except KeyError:
            self._template_cache[format_str] = ret = Template(self, format_str)
            return ret

    def list_senses(self,
        form:Optional[str] = None,
    ):
        return super().list_senses(form or '')
    
    def list_all_scripts(self) -> List[str]:
        return super().list_all_scripts()

    def make_hsdataset(
        self,
        inputs:List[str],
        batch_size:int = 128, 
        window_size:int = 8, 
        num_workers:int = 1, 
        dropout:float = 0, 
        token_filter:Callable[[str, str], bool] = None, 
        split_ratio:float = 0, 
        separate_default_morpheme:bool = False,
        seed:int = 0,
    ):
        return super().make_hsdataset(inputs, batch_size, window_size, num_workers, dropout, token_filter, split_ratio, separate_default_morpheme, seed)


def extract_substrings(
    text:str,
    min_cnt: int = 2,
    min_length: int = 2,
    max_length: int = 32,
    longest_only: bool = True,
    stop_chr: Optional[str] = None,
) -> List[Tuple[str, int]]:
    if not text:
        return []
    if min_cnt <= 0:
        raise ValueError('min_cnt must be greater than 0')
    if min_length <= 0:
        raise ValueError('min_length must be greater than 0')
    if max_length < min_length:
        raise ValueError('max_length must be greater than or equal to min_length')
    return _kiwipiepy._extract_substrings(text, min_cnt, min_length, max_length, longest_only, stop_chr or '\x00')
