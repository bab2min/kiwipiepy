'''
const 모듈은 kiwipiepy에서 사용되는 주요 상수값들을 모아놓은 모듈입니다.
'''

from enum import IntEnum

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
    URL = 1 << 0
    """ 인터넷 주소 형태의 텍스트를 W_URL이라는 태그로 추출합니다. """
    EMAIL = 1 << 1
    """ 이메일 주소 형태의 텍스트를 W_EMAIL이라는 태그로 추출합니다. """
    HASHTAG = 1 << 2
    """ 해시태그(#해시태그) 형태의 텍스트를 W_HASHTAG라는 태그로 추출합니다. """
    MENTION = 1 << 3
    """
    멘션(@멘션) 형태의 텍스트를 W_MENTION이라는 태그로 추출합니다.
    
    .. versionadded:: 0.8.2
    """
    SERIAL = 1 << 4
    """
    일련번호 형태의 텍스트를 W_SERIAL이라는 태그로 추출합니다.
    
    .. versionadded:: 0.14.0
    """
    ALL = URL | EMAIL | HASHTAG | MENTION | SERIAL
    """ URL, EMAIL, HASHTAG, MENTION, SERIAL을 모두 사용합니다. """
    NORMALIZING_CODA = 1 << 16
    """ '먹었엌ㅋㅋ'처럼 받침이 덧붙어서 분석에 실패하는 경우, 받침을 분리하여 정규화합니다. """
    JOIN_NOUN_PREFIX = 1 << 17
    """
    명사의 접두사를 분리하지 않고 결합합니다. 풋/XPN 사과/NNG -> 풋사과/NNG 

    .. versionadded:: 0.11.0
    """
    JOIN_NOUN_SUFFIX = 1 << 18
    """
    명사의 접미사를 분리하지 않고 결합합니다. 사과/NNG 들/XSN -> 사과들/NNG
    
    .. versionadded:: 0.11.0
    """
    JOIN_VERB_SUFFIX = 1 << 19
    """
    동사 파생접미사를 분리하지 않고 결합합니다. 사랑/NNG 하/XSV 다/EF -> 사랑하/VV 다/EF

    .. versionadded:: 0.11.0
    """
    JOIN_ADJ_SUFFIX = 1 << 20
    """
    형용사 파생접미사를 분리하지 않고 결합합니다. 매콤/XR 하/XSA 다/EF -> 매콤하/VA 다/EF

    .. versionadded:: 0.11.0
    """
    JOIN_V_SUFFIX = JOIN_VERB_SUFFIX | JOIN_ADJ_SUFFIX
    """
    동사/형용사형 파생접미사를 분리하지 않고 결합합니다.

    .. versionadded:: 0.11.0
    """
    JOIN_AFFIX = JOIN_NOUN_PREFIX | JOIN_NOUN_SUFFIX | JOIN_V_SUFFIX
    """
    모든 접두사/접미사를 분리하지 않고 결합합니다.

    .. versionadded:: 0.11.0
    """
