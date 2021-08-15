#pragma once

#define DOC_KO

#define DOC_SIGNATURE_EN(name, signature, en) PyDoc_STRVAR(name, signature "\n--\n\n" en)
#define DOC_VARIABLE_EN(name, en) PyDoc_STRVAR(name, en)
#ifdef DOC_KO
#define DOC_SIGNATURE_EN_KO(name, signature, en, ko) PyDoc_STRVAR(name, signature "\n--\n\n" ko)
#define DOC_VARIABLE_EN_KO(name, en, ko) PyDoc_STRVAR(name, ko)
#else
#define DOC_SIGNATURE_EN_KO(name, signature, en, ko) PyDoc_STRVAR(name, signature "\n--\n\n" en)
#define DOC_VARIABLE_EN_KO(name, en, ko) PyDoc_STRVAR(name, en)
#endif

DOC_SIGNATURE_EN_KO(Kiwi__doc__, 
    "Kiwi(self, num_workers=0, model_path=None, options=kiwipiepy.Option.DEFAULT, integrate_allomorph=True, load_default_dict=True)",
    u8R""()"",
    u8R""(Kiwi 클래스는 실제 형태소 분석을 수행하는 kiwipiepy 모듈의 핵심 클래스입니다.

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
    True일 경우 음운론적 이형태를 통합하여 출력합니다. /아/와 /어/나 /았/과 /었/ 같이 앞 모음의 양성/음성에 따라 형태가 바뀌는 어미들을 하나로 통합하여 출력합니다.
load_default_dict: bool
    True일 경우 인스턴스 생성시 자동으로 기본 사전을 불러옵니다. 기본 사전은 위키백과와 나무위키에서 추출된 고유 명사 표제어들로 구성되어 있습니다.
)"");

DOC_SIGNATURE_EN_KO(Kiwi_add_user_word__doc__, 
    "add_user_word(self, word, tag='NNP', score=0)",
    u8R""(add custom word into model)"",
    u8R""(현재 모델에 사용자 정의 단어를 추가합니다.

Parameters
----------
word: str
    추가할 단어
tag: str
    추가할 단어의 품사 태그
score: float
    추가할 단어의 가중치 점수. 
    해당 형태에 부합하는 형태소 조합이 여러 개가 있는 경우, 이 점수가 높을 단어가 더 우선권을 가집니다.
)"");

DOC_SIGNATURE_EN_KO(Kiwi_load_user_dictionary__doc__,
    "load_user_dictionary(self, dict_path)",
    u8R""(load custom dictionary file into model)"",
    u8R""(사용자 정의 사전을 읽어옵니다. 사용자 정의 사전 파일은 UTF-8로 인코딩된 텍스트 파일이어야 합니다.
사용자 정의 사전으로 추가된 단어의 개수를 반환합니다.
    
Parameters
----------
dict_path: str
    사용자 정의 사전 파일의 경로
)"");

DOC_SIGNATURE_EN_KO(Kiwi_extract_words__doc__,
    "extract_words(self, texts, min_cnt=10, max_word_len=10, min_score=0.25, pos_threshold=-3, lm_filter=True)",
    u8R""(extract words from corpus)"",
    u8R""(말뭉치로부터 새로운 단어를 추출합니다. 
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
)"");

DOC_SIGNATURE_EN_KO(Kiwi_extract_filter_words__doc__,
    "extract_filter_words(self, texts, min_cnt=10, max_word_len=10, min_score=0.25, pos_threshold=-3, lm_filter=True)",
    u8R""(extract words from corpus and filter the results)"",
    u8R""(.. deprecated:: 0.10.0
    이 메소드의 기능은 `kiwipiepy.Kiwi.extract_words`로 통합되었습니다. 
    현재 이 메소드를 호출하는 것은 `kiwipiepy.Kiwi.extract_words`를 호출하는 것과 동일하게 처리됩니다.
)"");

DOC_SIGNATURE_EN_KO(Kiwi_extract_add_words__doc__,
    "extract_add_words(self, texts, min_cnt=10, max_word_len=10, min_score=0.25, pos_score=-3, lm_filter=True)",
    u8R""(extract words from corpus and add them into model)"",
    u8R""(말뭉치로부터 새로운 단어를 추출하고 새로운 명사에 적합한 결과들만 추려냅니다. 그리고 그 결과를 현재 모델에 자동으로 추가합니다.

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
)"");

DOC_SIGNATURE_EN_KO(Kiwi_perform__doc__,
    "perform(self, texts, top_n=1, match_options=kiwipiepy.Match.ALL, min_cnt=10, max_word_len=10, min_score=0.25, pos_score=-3)",
    u8R""(extract_add_words + prepare + analyze)"",
    u8R""(현재 모델의 사본을 만들어
`kiwipiepy.Kiwi.extract_add_words`메소드로 말뭉치에서 단어를 추출하여 추가하고, `kiwipiepy.Kiwi.analyze`로 형태소 분석을 실시합니다.
이 메소드 호출 후 모델의 사본은 파괴되므로, 말뭉치에서 추출된 단어들은 다시 모델에서 제거되고, 메소드 실행 전과 동일한 상태로 돌아갑니다.

.. versionchanged:: 0.10.0
    입력을 단순히 문자열의 리스트로 주고, 분석 결과 역시 별도의 `receiver`로 받지 않고 바로 메소드의 리턴값으로 받게 변경되었습니다.
    자세한 내용은 https://bab2min.github.io/kiwipiepy/v0.10.0/kr/#0100 를 확인해주세요.

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
)"");

DOC_SIGNATURE_EN_KO(Kiwi_set_cutoff_threshold__doc__,
    "set_cutoff_threshold(self, threshold)",
    u8R""()"",
    u8R""(Beam 탐색 시 미리 제거할 후보의 점수 차를 설정합니다. 이 값이 클 수록 더 많은 후보를 탐색하게 되므로 분석 속도가 느려지지만 정확도가 올라갑니다.
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
)"");

DOC_SIGNATURE_EN_KO(Kiwi_prepare__doc__,
    "prepare(self)",
    u8R""(prepare the model to analyze text)"",
    u8R""(.. deprecated:: 0.10.0
    0.10.0버전부터 내부적으로 prepare()가 필요한 순간에 스스로 처리를 하도록 변경되어서 이제 이 메소드를 직접 호출할 필요가 없습니다.
    차기 버전에서 제거될 예정입니다.
    
)"");

DOC_SIGNATURE_EN_KO(Kiwi_get_option__doc__,
    "get_option(self, option)",
    u8R""(get option value)"",
    u8R""(현재 모델의 설정값을 가져옵니다.

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
)"");

DOC_SIGNATURE_EN_KO(Kiwi_set_option__doc__,
    "set_option(self, option, value)",
    u8R""(set option value)"",
    u8R""(현재 모델의 설정값을 변경합니다.

.. deprecated:: 0.10.0
    차기 버전에서 제거될 예정입니다. 
    이 메소드 대신 `Kiwi.integrate_allomorph`값을 직접 수정하십시오.

Parameters
----------
option: kiwipiepy.Option
    변경할 옵션의 열거값. 현재는 `kiwipiepy.Option.INTEGRATE_ALLOMORPH`만 지원합니다.
value: int
    0으로 설정할 경우 해당 옵션을 해제, 0이 아닌 값으로 설정할 경우 해당 옵션을 설정합니다.
)"");

DOC_SIGNATURE_EN_KO(Kiwi_analyze__doc__,
    "analyze(self, text, top_n=1, match_options=kiwipiepy.Match.ALL)",
    u8R""(analyze text and return top_n results)"",
    u8R""(형태소 분석을 실시합니다.

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

Returns
-------
result: List[Tuple[List[kiwipiepy.Token], float]]
    text를 str으로 준 경우.
    분석 결과는 최대 `top_n`개 길이의 리스트로 반환됩니다. 리스트의 각 항목은 `(분석 결과, 분석 점수)`로 구성된 튜플입니다. 
    `분석 결과`는 `kiwipiepy.Token`의 리스트로 구성됩니다.

results: Iterable[List[Tuple[List[kiwipiepy.Token], float]]]
    text를 Iterable[str]으로 준 경우.
    반환값은 iterator로 주어집니다. iterator가 차례로 반환하는 분석결과 값은 입력으로 준 text의 순서와 동일합니다.
)"");


DOC_VARIABLE_EN_KO(Kiwi_version__doc__,
    u8R""()"",
    u8R""(Kiwi의 버전을 반환합니다. 
.. deprecated:: 0.10.0
    차기 버전에서 제거될 예정입니다. 대신 `kiwipiepy.__version__`을 사용하십시오.
)""
);

DOC_VARIABLE_EN_KO(Kiwi_cutoff_threshold__doc__,
    u8R""()"",
    u8R""(.. versionadded:: 0.10.0

Beam 탐색 시 미리 제거할 후보의 점수 차를 설정합니다. 이 값이 클 수록 더 많은 후보를 탐색하게 되므로 분석 속도가 느려지지만 정확도가 올라갑니다.
반대로 이 값을 낮추면 더 적은 후보를 탐색하여 속도가 빨라지지만 정확도는 낮아집니다. 초기값은 5입니다.
)""
);

DOC_VARIABLE_EN_KO(Kiwi_integrate_allomorph__doc__,
    u8R""()"",
    u8R""(.. versionadded:: 0.10.0

True일 경우 음운론적 이형태를 통합하여 출력합니다. /아/와 /어/나 /았/과 /었/ 같이 앞 모음의 양성/음성에 따라 형태가 바뀌는 어미들을 하나로 통합하여 출력합니다.
)""
);

DOC_VARIABLE_EN_KO(Kiwi_num_workers__doc__,
    u8R""()"",
    u8R""(.. versionadded:: 0.10.0

병렬처리시 사용할 스레드의 개수입니다. (읽기 전용)
)""
);

DOC_SIGNATURE_EN_KO(Token__doc__,
    "Token(self)",
    u8R""()"",
    u8R""(Token은 분석 결과 얻어진 형태소 정보를 담는 데이터 클래스입니다. (`form`, `tag`, `start`, `len`) 형태의 길이 4의 `tuple`로 변환 가능합니다.)""
);

DOC_VARIABLE_EN_KO(Token_form__doc__,
    u8R""()"",
    u8R""(형태소의 형태)""
);

DOC_VARIABLE_EN_KO(Token_tag__doc__,
    u8R""()"",
    u8R""(형태소의 품사 태그)""
);

DOC_VARIABLE_EN_KO(Token_start__doc__,
    u8R""()"",
    u8R""(형태소의 입력 텍스트 내 시작 위치 (문자 단위))""
);

DOC_VARIABLE_EN_KO(Token_end__doc__,
    u8R""()"",
    u8R""(형태소의 입력 텍스트 내 끝 위치 (문자 단위))""
);

DOC_VARIABLE_EN_KO(Token_len__doc__,
    u8R""()"",
    u8R""(형태소의 입력 텍스트 내 차지 길이 (문자 단위))""
);

DOC_VARIABLE_EN_KO(Token_id__doc__,
    u8R""()"",
    u8R""(형태소의 내부 고유 ID)""
);
