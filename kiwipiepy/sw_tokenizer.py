'''
.. versionadded:: 0.15.1

`sw_tokenizer` 모듈은 서브워드 토크나이저와 관련된 클래스를 제공합니다.
'''

import re
import itertools
from typing import Callable, List, Optional, Tuple, Union, Iterable, Dict, Any
from dataclasses import dataclass
import warnings

import tqdm

from _kiwipiepy import _SwTokenizer

from kiwipiepy import Kiwi, Token

@dataclass
class SwTokenizerConfig:
    '''
    서브워드 토크나이저의 설정을 관리하는 데이터 클래스입니다.
    '''
    
    lowercase:bool = False
    '''토큰화에 앞서 대문자들을 전부 소문자로 정규화합니다.'''
    split_chinese:bool = True
    '''토큰화에 앞서 모든 한자를 한 글자씩 분리합니다.'''
    whole_word_unk:bool = False
    '''토크나이저의 어휘 집합에 속하지 않은 글자가 하나라도 포함된 어절 전체를 UNK토큰으로 지정합니다.'''
    integrate_allomorph: bool = True
    '''이형태를 통합합니다. (아직 미지원)'''
    split_punct: bool = True
    '''토큰화에 앞서 구두점을 분리합니다.'''
    simple_tag:bool = True
    '''세부 품사태그(예: VV, VX, VA) 대신 대표 품사태그(예: V)를 사용합니다. '''
    split_verb:bool = True
    '''서브워드로 분리가능한 동사가 있다면 더 작은 단위로 분리합니다. (예: 뛰어가다 -> 뛰/V 어/E 가/V 다/E)'''
    split_eomi:bool = True
    '''서브워드로 분리가능한 어미가 있다면 더 작은 단위로 분리합니다. (예: 어요 -> 어/E 요/J)'''
    use_glue_token:bool = True
    '''단어를 이어붙이는 특수 토큰인 Glue토큰을 사용합니다.'''
    use_newline_token:bool = False
    '''줄바꿈 문자를 토큰으로 사용합니다. False일 경우 공백으로 취급하여 무시합니다.
    줄바꿈 문자로는 fallback_byte 토큰의 10번째 바이트 토큰(\n)을 사용합니다.
    '''
    strict:bool = False
    '''토큰화 수행시 원문을 완벽하게 복원하도록 합니다. (아직 미지원)'''
    fallback_hangul:bool = True
    '''토크나이저의 어휘 집합에 속하지 않은 한글이 있다면 초성+중성 / 종성의 형태로 분해하여 처리합니다.'''
    fallback_byte:bool = False
    '''토크나이저의 어휘 집합에 속하지 않은 문자가 있다면 UTF8 Byte로 변환하여 처리합니다.'''

    unk_token:str = "[UNK]"
    cls_token:str = None
    sep_token:str = None
    pad_token:str = None
    mask_token:str = None
    bos_token:str = None
    eos_token:str = None

    additional: Any = None

class TrainerCallback:
    '''
    서브워드 토크나이저를 학습할 때 진행상황을 보고받기 위한 콜백 클래스입니다.
    이 클래스를 상속 받아서 필요한 기능을 구현한 뒤 `SwTokenizer.train` 함수의 `callback` 인자로 넘겨 사용할 수 있습니다.
    '''
    def begin_tokenization(self, num_processed_lines:int):
        '''
이 메소드는 학습을 위해 형태소 분석 작업이 시작됐을 때 호출됩니다.

Parameters
----------
num_processed_lines: int
    처리가 완료된 줄 수
        '''

    def proc_tokenization(self, num_processed_lines:int):
        '''
이 메소드는 형태소 분석 작업이 진행 중일 때 호출됩니다.
인자는 `begin_tokenization`과 동일합니다.
        '''

    def end_tokenization(self, num_processed_lines:int):
        '''
이 메소드는 형태소 분석 작업이 완료되었을 때 호출됩니다.
인자는 `begin_tokenization`과 동일합니다.
        '''

    def begin_reduction(self, n_tkn:int, iteration:int, cur_vocab_size:int, unigram_loss:float):
        '''
이 메소드는 형태소 분석 작업 완료 후 토큰 후보를 추리는 작업이 시작됐을 때 호출됩니다.

Parameters
----------
n_tkn: int
    현재 학습 중인 토크나이저의 순번

iteration: int
    현재 완료된 반복 횟수

cur_vocab_size: int
    현재 토큰 집합의 크기

unigram_loss: float
    손실함수의 현재 값
        '''

    def proc_reduction(self, n_tkn:int, iteration:int, cur_vocab_size:int, unigram_loss:float):
        '''
이 메소드는 토큰 후보를 추리는 작업이 진행 중일 때 호출됩니다.
인자는 `begin_reduction`과 동일합니다.
        '''

    def end_reduction(self, n_tkn:int, iteration:int, cur_vocab_size:int, unigram_loss:float):
        '''
이 메소드는 토큰 후보를 추리는 작업이 완료되었을 때 호출됩니다.
인자는 `begin_reduction`과 동일합니다.
        '''

class _ProgressShower(TrainerCallback):
    def __init__(self, file, total=None, iterations=None):
        super().__init__()
        self._file = file
        self._bar = None
        self._total = total
        self._iterations = iterations
    
    def __del__(self):
        if self._bar:
            self._bar.close()

    def begin_tokenization(self, num_processed_lines: int):
        self._bar = tqdm.tqdm(itertools.repeat(None), desc="Tokenizing", file=self._file, total=self._total)
        self._last_proc_lines = 0
    
    def proc_tokenization(self, num_processed_lines: int):
        self._bar.update(num_processed_lines - self._last_proc_lines)
        self._last_proc_lines = num_processed_lines
    
    def end_tokenization(self, num_processed_lines: int):
        self._bar.update(num_processed_lines - self._last_proc_lines)
        self._last_proc_lines = num_processed_lines
        self._bar.close()
        self._bar = None
    
    def begin_reduction(self, n_tkn:int, iteration: int, cur_vocab_size: int, unigram_loss: float):
        self._bar = tqdm.tqdm(itertools.repeat(None), desc=f"Reducing #{n_tkn+1}", file=self._file, total=self._iterations)
        self._bar.write(f"Iteration: {iteration} VocabSize: {cur_vocab_size} Loss: {-unigram_loss:.4f}")
        self._bar.set_postfix(dict(vocab_size=cur_vocab_size, loss=-unigram_loss))
        self._last_iteration = iteration
    
    def proc_reduction(self, n_tkn:int, iteration: int, cur_vocab_size: int, unigram_loss: float):
        self._bar.write(f"Iteration: {iteration} VocabSize: {cur_vocab_size} Loss: {-unigram_loss:.4f}")
        self._bar.update(iteration - self._last_iteration)
        self._bar.set_postfix(dict(vocab_size=cur_vocab_size, loss=-unigram_loss))
        self._last_iteration = iteration

    def end_reduction(self, n_tkn:int, iteration: int, cur_vocab_size: int, unigram_loss: float):
        self._bar.close()
        self._bar = None
        print(f"Finished. Iteration: {iteration} VocabSize: {cur_vocab_size} Loss: {-unigram_loss:.4f}", file=self._file)

SPECIAL_TOKEN_NAMES = ['unk', 'cls', 'sep', 'mask', 'pad', 'bos', 'eos']

class SwTokenizer(_SwTokenizer):
    '''
형태소 분석기 Kiwi를 기반으로 한 서브워드 토크나이저(Subword Tokenizer)를 제공하는 클래스입니다.
일반 서브워드 토크나이저와는 달리 한국어의 형태소를 고려하며 분할을 수행한다는 특징이 있습니다.

Parameters
----------
path: str
    불러들일 토크나이저 파일의 경로입니다.

kiwi: Kiwi
    형태소 분석에 사용될 Kiwi 형태소 분석기의 인스턴스입니다.
    생략하거나 None으로 줄 경우 내부에서 자체적으로 Kiwi 인스턴스를 생성하여 사용합니다.

num_workers: int
    멀티스레딩 시 사용할 스레드의 개수입니다.
    `kiwi`를 생략한 경우에만 유효합니다.
    `kiwi`를 지정한 경우 `kiwi.num_workers` 값에 의해 멀티스레딩을 수행합니다.
    '''

    def __init__(self,
        path: str,
        kiwi: Optional[Kiwi] = None,
        num_workers: Optional[int] = None,
    ):
        if kiwi is None:
            kiwi = Kiwi(num_workers=num_workers, load_multi_dict=False)
        elif num_workers is not None:
            raise ValueError("You cannot specify `num_workers` value if you give `kiwi` value.")

        if not isinstance(kiwi, Kiwi):
            raise ValueError("`kiwi` must be an instance of `Kiwi`.")

        super().__init__(kiwi, path)

        self._space_tolerance = (self.config.additional.get('space_tolerance') if isinstance(self.config.additional, dict) else None) or 0
    
    def encode(self, 
        text: Union[str, Iterable[str]],
        return_offsets: bool = False,
    ) -> Union[List[int], Tuple[List[int], List[Tuple[int, int]]]]:
        '''
주어진 텍스트를 토큰화하여 token id의 리스트로 반환합니다.

Parameters
----------
text: Union[str, Iterable[str]]
    분할할 텍스트, 혹은 텍스트의 iterable
    이 인자를 단일 str로 줄 경우, 싱글스레드에서 처리하며
    str의 Iterable로 줄 경우, 멀티스레드로 분배하여 처리합니다.

return_offsets: bool
    True일 경우 각 토큰들의 텍스트 상의 시작지점 및 끝지점이 함께 반환됩니다.

Returns
-------
token_ids: List[int]
    `text`를 단일 `str`으로 주고 `return_offsets = False`인 경우.
    token id의 목록을 list로 반환합니다.

iterable_of_token_ids: Iterable[List[int]]
    `text`를 `Iterable[str]`으로 주고 `return_offsets = False`인 경우.
    token id의 list를 나열하는 Iterable을 반환합니다.

token_ids_and_offsets: Tuple[List[int], List[Tuple[int, int]]]
    `text`를 단일 `str`으로 주고 `return_offsets = True`인 경우.
    token id의 리스트와 각 토큰들의 시작지점과 끝지점(문자 단위)을 나타내는 tuple의 리스트를 반환합니다.

iterable_of_token_ids_and_offsets: Iterable[Tuple[List[int], List[Tuple[int, int]]]]
    `text`를 `Iterable[str]`으로 주고 `return_offsets = True`인 경우.

Notes
-----
멀티스레딩은 SwTokenizer를 생성할 때 준 kiwi에 의해 수행됩니다.
따라서 kiwi 인자로 준 Kiwi 인스턴스의 num_workers 값에 따라 동시에 수행되는 작업의 개수가 결정됩니다.

```python
from kiwipiepy import Kiwi
from kiwipiepy.sw_tokenizer import SwTokenizer
kiwi = Kiwi(num_workers=1)
tokenizer = SwTokenizer('some_tokenizer.json', kiwi=kiwi)
tokenizer.encode(...) 
# kiwi.num_workers가 1이기 때문에 1개 스레드에서만 작업이 처리됨
tokenizer = SwTokenizer('some_tokenizer.json', num_workers=1)
tokenizer.encode(...) 
# 위와 동일하게 1개 스레드 사용

kiwi = Kiwi(num_workers=8)
tokenizer = SwTokenizer('some_tokenizer.json', kiwi=kiwi)
tokenizer.encode(...) 
# kiwi.num_workers가 8이기 때문에 8개 스레드에서 작업이 처리됨
tokenizer = SwTokenizer('some_tokenizer.json', num_workers=8)
tokenizer.encode(...) 
# 위와 동일하게 8개 스레드 사용
```
        '''
        self.kiwi.space_tolerance = self._space_tolerance
        return super().encode(text, return_offsets)
    
    def encode_from_morphs(self, 
        morphs: Iterable[Union[Tuple[str, str, bool], Tuple[str, str]]],
        return_offsets: bool = False,
    ) -> List[int]:
        '''
이미 형태소 분석이 완료된 결과를 토큰화하여 token id의 리스트로 변환합니다.

Parameters
----------
morphs: Iterable[Union[Tuple[str, str, bool], Tuple[str, str]]]
    `(형태, 품사태그, 왼쪽의 공백 여부)` 혹은 `(형태, 품사태그)`로 구성된 tuple의 리스트
    `왼쪽의 공백 여부`는 생략 가능하며 이 경우 `False`로 처리됩니다.

return_offsets: bool
    True일 경우 각 토큰들의 형태소 상의 시작지점 및 끝지점이 함께 반환됩니다.
    
Returns
-------
token_ids: List[int]
    `return_offsets = False`인 경우.
    token id의 목록을 list로 반환합니다.

token_ids_and_offsets: Tuple[List[int], List[Tuple[int, int]]]
    `return_offsets = True`인 경우.
    token id의 리스트와 각 토큰들의 시작지점과 끝지점(형태소 단위)을 나타내는 tuple의 리스트를 반환합니다.

        '''
        self.kiwi.space_tolerance = self._space_tolerance
        return super().encode_from_morphs(morphs, return_offsets)

    def tokenize_encode(self, 
        text: Union[str, Iterable[str]],
        return_offsets: bool = False,
    ) -> Union[Tuple[List[Token], List[int]], Tuple[List[Token], List[int], List[Tuple[int, int]]]]:
        '''
주어진 텍스트의 형태소 분석 결과 및 토큰화 결과를 함께 반환합니다.

Parameters
----------
text: Union[str, Iterable[str]]
    분할할 텍스트, 혹은 텍스트의 iterable
    이 인자를 단일 str로 줄 경우, 싱글스레드에서 처리하며
    str의 Iterable로 줄 경우, 멀티스레드로 분배하여 처리합니다.

return_offsets: bool
    True일 경우 각 토큰들의 텍스트 상의 시작지점 및 끝지점이 함께 반환됩니다.

Returns
-------
morphs_and_token_ids: Tuple[List[Token], List[int]]
    `return_offsets = False`인 경우.
    형태소 분석 결과와 token id의 목록을 tuple로 반환합니다.

morphs_token_ids_and_offsets: Tuple[List[Token], List[int], List[Tuple[int, int]]]
    `return_offsets = True`인 경우.
    형태소 분석 결과와 token id의 목록, 각 토큰들의 시작지점과 끝지점(형태소 단위)을 나타내는 tuple의 리스트를 반환합니다.

Notes
-----

        '''
        self.kiwi.space_tolerance = self._space_tolerance

        def _refine(res):
            morphs, *etc = res
            return (morphs[0][0], *etc)

        ret = super().tokenize_encode(text, return_offsets)
        if isinstance(text, str):
            return _refine(ret)
        else:
            return map(_refine, ret)

    def decode(self,
        ids: Iterable[int],
        ignore_errors: bool = True,
    ) -> str:
        '''
token id의 리스트를 다시 조합하여 텍스트로 변환합니다.

Parameters
----------
ids: Iterable[int]
    token id의 리스트
ignore_errors: bool
    token을 유니코드 텍스트로 복원할 때 발생하는 오류를 무시할지 설정합니다.
    기본값은 True, 이 경우 오류가 난 부분은 대체문자로 대체됩니다.
    False인 경우 오류가 났을 때 예외를 발생시킵니다.
    
Returns
-------
decoded: str
    다시 조합된 텍스트

Notes
-----
`encode`시 정규화 과정에서 공백 문자나 구두점, 한자 등의 띄어쓰기 변경이 발생하므로
`encode`한 결과를 `decode`한다고 해서 항상 동일한 결과가 나오지는 않습니다.
        '''
        return super().decode(ids, ignore_errors)
    
    def save(self, path:str):
        '''
현재 토크나이저를 json 파일로 저장합니다.
Parameters
----------
path: str
    저장할 파일의 경로

        '''
        return super().save(path)

    @property
    def vocab(self) -> Dict[str, int]:
        '''
토크나이저에 속한 어휘 집합을 dict로 반환합니다. (읽기 전용)
        '''
        try:
            return self._cached_vocab
        except AttributeError:
            self._cached_vocab = super()._vocab
            return self._cached_vocab

    @property
    def id2vocab(self) -> List[str]:
        try:
            return self._cached_id2vocab
        except AttributeError:
            ret = [None] * len(self)
            for k, v in self.vocab.items():
                ret[v] = k
            self._cached_id2vocab = ret
            return self._cached_id2vocab

    @property
    def config(self) -> SwTokenizerConfig:
        '''
토크나이저의 설정값을 담은 `SwTokenizerConfig` 데이터 클래스를 반환합니다. (읽기 전용)
        '''
        try:
            return self._cached_config
        except:
            self._cached_config = SwTokenizerConfig(**super()._config)
            return self._cached_config

    @property
    def kiwi(self) -> Kiwi:
        '''
토크나이저가 사용하는 Kiwi 인스턴스를 반환합니다. (읽기 전용)
        '''
        return super()._kiwi

    def __repr__(self) -> str:
        return super().__repr__()

    @property
    def unk_token(self) -> Optional[str]:
        return self.config.unk_token
    
    @property
    def pad_token(self) -> Optional[str]:
        return self.config.pad_token

    @property
    def mask_token(self) -> Optional[str]:
        return self.config.mask_token

    @property
    def cls_token(self) -> Optional[str]:
        return self.config.cls_token
    
    @property
    def sep_token(self) -> Optional[str]:
        return self.config.sep_token
    
    @property
    def bos_token(self) -> Optional[str]:
        return self.config.bos_token
    
    @property
    def eos_token(self) -> Optional[str]:
        return self.config.eos_token

    @property
    def unk_token_id(self) -> Optional[int]:
        t = self.config.unk_token
        if t is None: return None
        return self.vocab[t]
    
    @property
    def pad_token_id(self) -> Optional[int]:
        t = self.config.pad_token
        if t is None: return None
        return self.vocab[t]

    @property
    def mask_token_id(self) -> Optional[int]:
        t = self.config.mask_token
        if t is None: return None
        return self.vocab[t]

    @property
    def cls_token_id(self) -> Optional[int]:
        t = self.config.cls_token
        if t is None: return None
        return self.vocab[t]
    
    @property
    def sep_token_id(self) -> Optional[int]:
        t = self.config.sep_token
        if t is None: return None
        return self.vocab[t]
    
    @property
    def bos_token_id(self) -> Optional[int]:
        t = self.config.bos_token
        if t is None: return None
        return self.vocab[t]
    
    @property
    def eos_token_id(self) -> Optional[int]:
        t = self.config.eos_token
        if t is None: return None
        return self.vocab[t]

    @property
    def all_special_tokens(self):
        ret = []
        for t in SPECIAL_TOKEN_NAMES:
            v = getattr(self, t + '_token')
            if v is not None:
                ret.append(v)
        return ret

    @property
    def all_special_ids(self):
        ret = []
        for t in SPECIAL_TOKEN_NAMES:
            v = getattr(self, t + '_token_id')
            if v is not None:
                ret.append(v)
        return ret

    @staticmethod
    def train(
        save_path: Union[str, Iterable[str]],
        texts: Iterable[str],
        config: SwTokenizerConfig,
        vocab_size: Union[int, Iterable[int]],
        chr_coverage: float = 0.9995,
        prefix_min_cnt: int = 5,
        prefix_max_length: int = 15,
        strict_reduction: bool = False,
        remove_repetitive: bool = True,
        prevent_mixed_digit_tokens: bool = True,
        iterations: int = 1000,
        reduction_ratio: float = 0.1,
        kiwi: Optional[Kiwi] = None,
        num_workers: Optional[int] = None,
        show_progress: bool = True,
        total_texts: Optional[int] = None,
        callback : Optional[Union[TrainerCallback, List[TrainerCallback]]] = None,
    ) -> 'SwTokenizer':
        '''
주어진 텍스트로부터 유니그램 언어모델과 형태소 분석을 통합한 알고리즘을 통해 서브워드 토크나이저를 학습합니다. 

또한 목표하는 `vocab_size`가 서로 다른 여러 벌의 토크나이저를 한 번에 학습하는 것도 가능합니다.
이 경우 `save_path`를 str의 리스트로, `vocab_size`를 int의 리스트로 입력해야 합니다.

Parameters
----------
save_path: Union[str, Iterable[str]]
    학습된 토크나이저가 저장될 파일 이름을 지정합니다.

texts: Iterable[str]
    학습에 사용될 말뭉치 텍스트를 지정합니다.

config: SwTokenizerConfig
    토크나이저 설정을 지정합니다.

vocab_size: Union[int, Iterable[int]]
    토크나이저의 어휘 집합의 상한치를 지정합니다.

chr_coverage: float
    어휘 집합의 후보를 구성할 때 말뭉치에 등장한 글자 중 최대 얼마까지를 다룰지를 지정합니다.
    기본값은 0.9995로, 이 경우 말뭉치에 등장한 글자들 중 최대 99.95%를 다룰 수 있도록 어휘 집합을 구성합니다.
    이는 한자나 이모지, 일부 특수기호처럼 종류는 많지만 각각의 등장빈도가 낮은 글자를 배제하여 어휘 집합의 크기를 줄이는 데에 유용합니다.

prefix_min_cnt: int
    어휘 집합의 후보를 구성할 때 최소 몇 번 이상 등장한 접두어를 포함할지 지정합니다.
    기본값은 5로, 이 경우 말뭉치에서 최소 5번 이상 등장한 접두어들만이 어휘 집합의 후보로 들어갑니다.

prefix_max_length: int
    어휘 집합의 후보를 구성할 때 포함되는 접두어의 최대 길이를 지정합니다.
    기본값은 15로, 이 경우 최대 15글자로 구성되는 접두어까지 어휘 집합의 후보로 들어갈 수 있습니다.

prevent_mixed_digit_tokens: bool
    어휘 집합의 후보를 구성할 때 숫자와 다른 문자가 섞인 토큰을 배제합니다.
    기본값은 True로, 이 경우 "1분", "10시" 등의 표현은 항상 `1` `분`, `10` `시`처럼 분할됩니다.

strict_reduction: bool
    어휘 집합을 줄여나갈 때 한 번 배제된 후보가 다시는 사용되지 못하도록 엄격하게 설정합니다. 
    기본값은 False입니다.
    True일 경우, 배제된 후보를 다시 고려할 필요가 없으므로 학습 속도가 약간 빨라집니다.
    반면 False일 경우, 학습 속도가 약간 느려지지만 배제된 후보 중 일부를 상황에 따라 다시 후보에 넣을 수 있으므로
    더 좋은 어휘 집합을 찾아낼 가능성이 있습니다.

remove_repetitive: bool
    어휘 집합을 구성할때 특정 패턴이 여러번 반복되는 접두어(ex: 안녕안녕안녕)를 배제합니다.
    기본값은 True입니다.
    
iteration: int
    어휘 집합을 줄여나가는 과정을 최대 몇 번 반복할지 설정합니다.
    기본값은 1000이지만, 대체로 1000회보다 더 작은 횟수 안에 학습이 완료됩니다.

reduction_ratio: float 
    어휘 집합을 줄여나갈 비율을 설정합니다.
    기본값은 0.1으로, 이 경우 1회 반복시마다 어휘 집합 내의 어휘 중 최대 10%를 제거합니다.
    이 값이 클 수록 더 빨리 학습이 완료되지만, 너무 과감하게 어휘들을 제거하므로
    최종 생성된 어휘 집합의 품질이 낮을 수 있습니다.

kiwi: Kiwi
    학습에 사용할 Kiwi 형태소 분석기 인스턴스입니다.
    생략하거나 None으로 줄 경우 내부에서 자체적으로 Kiwi 인스턴스를 생성하여 사용합니다.
    
num_workers: int
    멀티스레딩 시 사용할 스레드의 개수입니다.
    `kiwi`를 생략한 경우에만 유효합니다.
    `kiwi`를 지정한 경우 `kiwi.num_workers` 값에 의해 멀티스레딩을 수행합니다.

show_progress: bool
    True일 경우 학습의 진행상황을 화면에 보여줍니다. 기본값은 True입니다.

total_texts: int
    `texts`의 전체 길이를 지정합니다. 이 값은 오직 `show_progress=True`일때만 유효하며
    전체 입력의 길이를 이용해 현재 학습이 얼마나 진행되었는지 보여주는 데에 쓰입니다.
    생략 시 `len(texts)`을 이용해 입력의 길이를 스스로 추정합니다.

callback: Union[TrainerCallback, List[TrainerCallback]]
    학습 진행상황을 보고 받을 callback 클래스의 인스턴스를 설정합니다.
    각 인스턴스들은 `TrainerCallback` 클래스를 상속 받은 클래스여야 합니다.

Returns
-------
tokenizer: SwTokenizer
    `save_path`와 `vocab_size`가 각각 `str`, `int`로 주어진 경우.
    학습이 완료된 후 얻어진 토크나이저를 반환합니다.
    또한 토크나이저는 `save_path`로 지정된 경로에 저장되어 있으니
    `SwTokenizer(save_path)`를 통해 새로 읽어들이는 것도 가능합니다.

tokenizers: List[SwTokenizer]
    `save_path`와 `vocab_size`가 각각 `Iterable[str]`, `Iterable[int]`로 주어진 경우.
    학습이 완료된 후 얻어진 토크나이저의 리스트를 반환합니다.
    또한 토크나이저는 `save_path`로 지정된 경로 각각에 저장되어 있으니
    `SwTokenizer(save_path[i])`를 통해 새로 읽어들이는 것도 가능합니다.

Notes
-----
```python
```
        '''
        if kiwi is None:
            kiwi = Kiwi(num_workers=num_workers, load_multi_dict=False)
        elif num_workers is not None:
            raise ValueError("You cannot specify `num_workers` value if you give `kiwi` value.")
                
        if not isinstance(kiwi, Kiwi):
            raise ValueError("`kiwi` must be an instance of `Kiwi`.")
        
        if not isinstance(config, SwTokenizerConfig):
            raise ValueError("`config` must be an instance of `SwTokenizerConfig`.")

        if callback is None:
            callback = []
        elif isinstance(callback, TrainerCallback):
            callback = [callback]
        else:
            callback = list(callback)
        
        if not all(isinstance(c, TrainerCallback) for c in callback):
            raise ValueError("`callback` must be an instance of `TrainerCallback`.")

        if show_progress:
            if total_texts is None:
                try: total_texts = len(texts)
                except: pass
            callback.insert(0, _ProgressShower(None if show_progress is True else show_progress, total_texts, iterations))

        single_target = False
        if isinstance(save_path, str):
            if not isinstance(vocab_size, int):
                raise ValueError("`save_path` should have the same number of elements to `vocab_size`.")
            single_target = True
            save_path = [save_path]
            vocab_size = [vocab_size]
        
        _SwTokenizer._train(
            save_path, 
            texts, 
            config, 
            vocab_size, 
            iterations,
            prefix_min_cnt,
            prefix_max_length,
            strict_reduction, 
            remove_repetitive, 
            prevent_mixed_digit_tokens,
            chr_coverage, 
            reduction_ratio, 
            kiwi, 
            callback,
        )
        if single_target:
            return SwTokenizer(save_path[0], kiwi)
        else:
            return [SwTokenizer(s, kiwi) for s in save_path]

