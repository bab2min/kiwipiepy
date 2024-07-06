from typing import Union, Optional, Tuple, NewType

POSTag = NewType('POSTag', str)

class Token:
    '''
    `Token`은 분석 결과 얻어진 형태소 정보를 담는 데이터 클래스입니다. (`form`, `tag`, `start`, `len`) 형태의 길이 4의 `tuple`로 변환 가능합니다.
    '''

    @property
    def form(self) -> str: 
        '''형태소의 형태'''
        ...

    @property
    def tag(self) -> POSTag:
        '''형태소의 품사 태그'''
        ...

    @property
    def start(self) -> int:
        '''형태소의 입력 텍스트 내 시작 위치 (문자 단위)'''
        ...

    @property
    def end(self) -> int:
        '''형태소의 입력 텍스트 내 끝 위치 (문자 단위)'''
        ...

    @property
    def span(self) -> Tuple[int, int]:
        '''.. versionadded:: 0.16.0
        
형태소의 입력 텍스트 내 시작 및 끝 위치 (문자 단위)'''
        ...

    @property
    def len(self) -> int:
        '''형태소의 입력 텍스트 내 차지 길이 (문자 단위)'''
        ...

    @property
    def id(self) -> int:
        '''형태소의 내부 고유 ID'''
        ...

    @property
    def word_position(self) -> int:
        '''.. versionadded:: 0.10.2

형태소의 입력 텍스트 내 어절 위치 (공백 기준, 문장별로 0부터 시작)'''
        ...

    @property
    def sent_position(self) -> int:
        '''.. versionadded:: 0.10.3

형태소의 입력 텍스트 내 문장 번호 (0부터 시작)'''
        ...

    @property
    def sub_sent_position(self) -> int:
        '''.. versionadded:: 0.14.0

형태소가 안긴 문장에 속한 경우, 현 문장 내의 안긴 문장 번호. (1부터 시작. 0일 경우 안긴 문장이 아님)'''
        ...

    @property
    def line_number(self) -> int:
        '''.. versionadded:: 0.10.3

형태소의 입력 텍스트 내 줄 번호 (0부터 시작)'''
        ...

    @property
    def base_form(self) -> str:
        '''.. versionadded:: 0.11.0

이형태의 경우 원본 형태소의 형태. 일반 형태소의 경우 `form`과 동일.'''
        ...

    @property
    def base_id(self) -> int:
        '''.. versionadded:: 0.11.0

이형태의 경우 원본 형태소의 고유 ID. 일반 형태소의 경우 `id`와 동일.'''
        ...


    @property
    def tagged_form(self) -> str:
        '''.. versionadded:: 0.11.1

form과 tag를 `형태/품사태그`꼴로 합쳐서 반환합니다.'''
        ...

    @property
    def form_tag(self) -> Tuple[str, str]:
        '''.. versionadded:: 0.15.2

(form, tag)를 반환합니다.'''
        ...

    @property
    def score(self) -> float:
        '''.. versionadded:: 0.12.0

현재 형태소의 언어 모델 상의 점수를 반환합니다.'''
        ...

    @property
    def regularity(self) -> Optional[bool]:
        '''.. versionadded:: 0.12.0

동/형용사가 규칙 활용하는 경우 True, 불규칙 활용하는 경우 False를 반환합니다.
그 외의 품사에 대해서는 None을 반환합니다.'''
        ...

    @property
    def lemma(self) -> str:
        '''.. versionadded:: 0.15.0

동/형용사의 경우 '-다'를 붙여서 형태소의 사전 표제형을 반환합니다.'''
        ...

    @property
    def typo_cost(self) -> float:
        '''.. versionadded:: 0.13.0

현재 형태소의 오타 교정 비용을 반환합니다.'''
        ...

    @property
    def raw_form(self) -> str:
        '''.. versionadded:: 0.13.0

텍스트 상에 실제로 등장한 형태. 오타가 교정된 경우 `form`과 다를 수 있음.'''
        ...

    @property
    def paired_token(self) -> int:
        '''.. versionadded:: 0.15.1

현재 형태소가 짝을 이루는 다른 형태소를 가지고 있다면 그 형태소의 index를 반환합니다. 없는 경우 -1을 반환합니다.'''
        ...

    @property
    def user_value(self):
        '''.. versionadded:: 0.16.0

사용자가 사전에 직접 형태소를 추가할 때 `user_value`로 입력한 값. 별도의 `user_value`를 입력하지 않은 경우 None을 반환합니다.'''
        ...

    @property
    def script(self):
        '''.. versionadded:: 0.18.0
        
해당 문자열이 어떤 언어 문자 집합에 속하는지를 나타냅니다. 문자 집합의 전체 목록에 대해서는 `Kiwi.list_all_scripts()`를 참조하세요.'''
