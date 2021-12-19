"""
.. include:: ./documentation.rst
"""
from typing import Optional
from enum import IntEnum
import warnings

from _kiwipiepy import _Kiwi, Token
from kiwipiepy._version import __version__
import kiwipiepy.utils as utils

class Option(IntEnum):
    """
    Kiwi 인스턴스 생성 시 사용 가능한 옵션 열거형. 
    bitwise or 연산으로 여러 개 선택하여 사용가능합니다.

    .. deprecated:: 0.10.0
        추후 버전에서 제거될 예정입니다.
    """

    LOAD_DEFAULT_DICTIONARY = 1
    """
    인스턴스 생성시 자동으로 기본 사전을 불러옵니다. 기본 사전은 위키백과와 나무위키에서 추출된 고유 명사 표제어들로 구성되어 있습니다.
    """
    INTEGRATE_ALLOMORPH = 2
    """
    음운론적 이형태를 통합하여 출력합니다. /아/와 /어/나 /았/과 /었/ 같이 앞 모음의 양성/음성에 따라 형태가 바뀌는 어미들을 하나로 통합하여 출력합니다.
    """
    DEFAULT = 3
    """
    Kiwi 생성시의 기본 옵션으로 LOAD_DEFAULT_DICTIONARY | INTEGRATE_ALLOMORPH 와 같습니다.
    """

class Match(IntEnum):
    """
    .. versionadded:: 0.8.0

    분석 시 특수한 문자열 패턴 중 어떤 것들을 추출할 지 선택할 수 있습니다.
    bitwise OR 연산으로 여러 개 선택하여 사용가능합니다.
    """
    URL = 1
    """
    인터넷 주소 형태의 텍스트를 W_URL이라는 태그로 추출합니다.
    """
    EMAIL = 2
    """
    이메일 주소 형태의 텍스트를 W_EMAIL이라는 태그로 추출합니다.
    """
    HASHTAG = 4
    """
    해시태그(#해시태그) 형태의 텍스트를 W_HASHTAG라는 태그로 추출합니다.
    """
    MENTION = 8
    """
    멘션(@멘션) 형태의 텍스트를 W_MENTION이라는 태그로 추출합니다.
    
    .. versionadded:: 0.8.2
    """
    ALL = 15
    """
    URL, EMAIL, HASHTAG, MENTION를 모두 사용합니다.
    """
    NORMALIZING_CODA = 65536
    """
    '먹었엌ㅋㅋ'처럼 받침이 덧붙어서 분석에 실패하는 경우, 받침을 분리하여 정규화합니다.
    """

del IntEnum

class Kiwi(_Kiwi):
    '''Kiwi 클래스는 실제 형태소 분석을 수행하는 kiwipiepy 모듈의 핵심 클래스입니다.

Parameters
----------
num_workers: int
    내부적으로 멀티스레딩에 사용할 스레드 개수. 0으로 설정시 시스템 내 가용한 모든 코어 개수만큼 스레드가 생성됩니다.
    멀티스레딩은 extract 계열 함수에서 단어 후보를 탐색할 때와 perform, async_analyze 함수 및 reader/receiver를 사용한 analyze 함수에서만 사용되며,
    단순 analyze는 단일스레드에서 돌아갑니다.
model_path: str
    읽어들일 모델 파일의 경로. 모델 파일의 위치를 옮긴 경우 이 값을 지정해주어야 합니다.
options: int
    Kiwi 생성시의 옵션을 설정합니다. 옵션에 대해서는 `kiwipiepy.Option`을 확인하십시오.
    .. deprecated:: 0.10.0
        차기 버전에서 제거될 예정입니다. `options` 대신 `integrate_allormoph` 및 `load_default_dict`를 사용해주세요.

integrate_allormoph: bool
    True일 경우 음운론적 이형태를 통합하여 출력합니다. /아/와 /어/나 /았/과 /었/ 같이 앞 모음의 양성/음성에 따라 형태가 바뀌는 어미들을 하나로 통합하여 출력합니다. 기본값은 True입니다.
load_default_dict: bool
    True일 경우 인스턴스 생성시 자동으로 기본 사전을 불러옵니다. 기본 사전은 위키백과와 나무위키에서 추출된 고유 명사 표제어들로 구성되어 있습니다. 기본값은 True입니다.
    '''

    def __init__(self, 
        num_workers:Optional[int] = None,
        model_path:Optional[str] = None,
        options:Optional[int] = None,
        integrate_allomorph:Optional[bool] = None,
        load_default_dict:Optional[bool] = None,
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

        super().__init__(
            num_workers=num_workers,
            model_path=model_path,
            integrate_allomorph=integrate_allomorph,
            load_default_dict=load_default_dict,
        )

    def add_user_word(self,
        word:str,
        tag:Optional[str] = 'NNP',
        score:Optional[float] = 0.,
    ):
        '''현재 모델에 사용자 정의 단어를 추가합니다.

Parameters
----------
word: str
    추가할 단어
tag: str
    추가할 단어의 품사 태그
score: float
    추가할 단어의 가중치 점수. 
    해당 형태에 부합하는 형태소 조합이 여러 개가 있는 경우, 이 점수가 높을 단어가 더 우선권을 가집니다.

Returns
-------
inserted: bool
    사용자 정의 단어가 정상적으로 삽입된 경우 True, 중복 단어 등의 이유로 삽입에 실패한 경우 False를 반환합니다.
        '''

        return super().add_user_word(word, tag, score)
    
    def load_user_dictionary(self,
        dict_path:str
    ):
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
    이 메소드는 0.10.0 버전에서 사용법이 일부 변경되었습니다. 자세한 내용은 https://bab2min.github.io/kiwipiepy/v0.10.0/kr/#0100 를 확인해주세요.

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
        min_cnt:int = 10,
        max_word_len:int = 10,
        min_score:float = 0.25,
        pos_score:float = -3.,
        lm_filter:bool = True,
    ):
        '''말뭉치로부터 새로운 단어를 추출하고 새로운 명사에 적합한 결과들만 추려냅니다. 그리고 그 결과를 현재 모델에 자동으로 추가합니다.

.. versionchanged:: 0.10.0
    이 메소드는 0.10.0 버전에서 사용법이 일부 변경되었습니다. 자세한 내용은 https://bab2min.github.io/kiwipiepy/v0.10.0/kr/#0100 를 확인해주세요.

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
        top_n:int = 1,
        match_options:int = Match.ALL,
        min_cnt:int = 10,
        max_word_len:int = 10,
        min_score:float = 0.25,
        pos_score:float = -3.,
        lm_filter:bool = True,
    ):
        '''현재 모델의 사본을 만들어
`kiwipiepy.Kiwi.extract_add_words`메소드로 말뭉치에서 단어를 추출하여 추가하고, `kiwipiepy.Kiwi.analyze`로 형태소 분석을 실시합니다.
이 메소드 호출 후 모델의 사본은 파괴되므로, 말뭉치에서 추출된 단어들은 다시 모델에서 제거되고, 메소드 실행 전과 동일한 상태로 돌아갑니다.

.. versionchanged:: 0.10.0
    입력을 단순히 문자열의 리스트로 주고, 분석 결과 역시 별도의 `receiver`로 받지 않고 바로 메소드의 리턴값으로 받게 변경되었습니다.
    자세한 내용은 https://bab2min.github.io/kiwipiepy/v0.10.0/kr/#0100 를 확인해주세요.

.. deprecated:: 0.10.1
    추후 버전에서 변경, 혹은 제거될 가능성이 있는 메소드입니다.

Parameters
----------
texts: Iterable[str]
    분석할 문자열의 리스트, 혹은 Iterable입니다.
top_n: int
    분석 결과 후보를 상위 몇 개까지 생성할 지 설정합니다.
match_options: int
    .. versionadded:: 0.8.0

    추출한 특수 문자열 패턴을 지정합니다. `kiwipiepy.Match`의 조합으로 설정할 수 있습니다.
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
        text,
        top_n:int = 1,
        match_options:int = Match.ALL,
        normalize_coda:bool = False
    ):
        '''형태소 분석을 실시합니다.

.. versionchanged:: 0.10.0
    이 메소드는 0.10.0 버전에서 사용법이 일부 변경되었습니다. 자세한 내용은 https://bab2min.github.io/kiwipiepy/v0.10.0/kr/#0100 를 확인해주세요.

Parameters
----------
text: Union[str, Iterable[str]]
    분석할 문자열입니다. 
    이 인자를 단일 str로 줄 경우, 싱글스레드에서 처리하며
    str의 Iterable로 줄 경우, 멀티스레드로 분배하여 처리합니다.
top_n: int
    분석 결과 후보를 상위 몇 개까지 생성할 지 설정합니다.
match_options: int
    
    .. versionadded:: 0.8.0
    
    추출한 특수 문자열 패턴을 지정합니다. `kiwipiepy.Match`의 조합으로 설정할 수 있습니다.
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
option: kiwipiepy.Option
    검사할 옵션의 열거값. 현재는 `kiwipiepy.Option.INTEGRATE_ALLOMORPH`만 지원합니다.

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
option: kiwipiepy.Option
    변경할 옵션의 열거값. 현재는 `kiwipiepy.Option.INTEGRATE_ALLOMORPH`만 지원합니다.
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
    
    @property
    def cutoff_threshold(self):
        '''.. versionadded:: 0.10.0

Beam 탐색 시 미리 제거할 후보의 점수 차를 설정합니다. 이 값이 클 수록 더 많은 후보를 탐색하게 되므로 분석 속도가 느려지지만 정확도가 올라갑니다.
반대로 이 값을 낮추면 더 적은 후보를 탐색하여 속도가 빨라지지만 정확도는 낮아집니다. 초기값은 5입니다.
        '''

        return self._cutoff_threshold
    
    @cutoff_threshold.setter
    def cutoff_threshold(self, v:float):
        self._cutoff_threshold = v
    
    @property
    def integrate_allomorph(self):
        '''.. versionadded:: 0.10.0

True일 경우 음운론적 이형태를 통합하여 출력합니다. /아/와 /어/나 /았/과 /었/ 같이 앞 모음의 양성/음성에 따라 형태가 바뀌는 어미들을 하나로 통합하여 출력합니다.
        '''

        return self._integrate_allomorph
    
    @integrate_allomorph.setter
    def integrate_allomorph(self, v:bool):
        self._integrate_allomorph = v
    
    @property
    def num_workers(self):
        '''.. versionadded:: 0.10.0

병렬처리시 사용할 스레드의 개수입니다. (읽기 전용)
        '''
        
        return self._num_workers
    
    def tokenize(self, 
        text, 
        match_options:int = Match.ALL,
        normalize_coda:bool = False,
    ):
        '''.. versionadded:: 0.10.2

간단하게 분석결과만 반환합니다.
Parameters
----------
text: Union[str, Iterable[str]]
    분석할 문자열입니다. 
    이 인자를 단일 str로 줄 경우, 싱글스레드에서 처리하며
    str의 Iterable로 줄 경우, 멀티스레드로 분배하여 처리합니다.
match_options: int
    추출한 특수 문자열 패턴을 지정합니다. `kiwipiepy.Match`의 조합으로 설정할 수 있습니다.
normalize_coda: bool
    True인 경우 '먹었엌ㅋㅋ'처럼 받침이 덧붙어서 분석에 실패하는 경우, 받침을 분리하여 정규화합니다.

Returns
-------
result: List[kiwipiepy.Token]
    text를 str으로 준 경우.
    `kiwipiepy.Token`의 리스트를 반환합니다.

results: Iterable[List[kiwipiepy.Token]]
    text를 Iterable[str]으로 준 경우.
    반환값은 iterator로 주어집니다. iterator가 차례로 반환하는 분석결과 값은 입력으로 준 text의 순서와 동일합니다.
        '''
        def _refine_result(results):
            return results[0][0]

        if isinstance(text, str):
            return _refine_result(self.analyze(text, top_n=1, match_options=match_options, normalize_coda=normalize_coda))
            
        return map(_refine_result, self.analyze(text, top_n=1, match_options=match_options, normalize_coda=normalize_coda))
