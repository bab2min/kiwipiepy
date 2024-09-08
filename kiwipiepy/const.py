'''
const 모듈은 kiwipiepy에서 사용되는 주요 상수값들을 모아놓은 모듈입니다.
'''

from enum import IntEnum

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
    EMOJI = 1 << 5
    """
    이모지 형태의 텍스트를 W_EMOJI라는 태그로 추출합니다.

    .. versionadded:: 0.18.0
    """
    ALL = URL | EMAIL | HASHTAG | MENTION | SERIAL | EMOJI
    """ URL, EMAIL, HASHTAG, MENTION, SERIAL, EMOJI을 모두 사용합니다. """
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
    JOIN_ADV_SUFFIX = 1 << 21
    """
    부사 파생접미사를 분리하지 않고 결합합니다. 요란/XR 히/XSM -> 요란히/MAG

    .. versionadded:: 0.15.0
    """
    SPLIT_COMPLEX = 1 << 22
    """
    더 잘게 분할 가능한 형태소를 모두 분할합니다. 고마움/NNG -> 고맙/VA-I 음/ETN

    .. versionadded:: 0.15.0
    """
    Z_CODA = 1 << 23
    """
    조사/어미에 덧붙은 받침을 Z_CODA 태그로 분리합니다. 했어욗 -> 하/VV 었/EP 어요/EF ㄳ/Z_CODA

    .. versionadded:: 0.15.0
    """
    COMPATIBLE_JAMO = 1 << 24
    """
    형태소 분석 결과 출력 시 첫가끝 자모를 호환용 자모로 변환합니다.

    .. versionadded:: 0.18.1
    """
    JOIN_V_SUFFIX = JOIN_VERB_SUFFIX | JOIN_ADJ_SUFFIX
    """
    동사/형용사형 파생접미사를 분리하지 않고 결합합니다.

    .. versionadded:: 0.11.0
    """
    JOIN_AFFIX = JOIN_NOUN_PREFIX | JOIN_NOUN_SUFFIX | JOIN_V_SUFFIX | JOIN_ADV_SUFFIX
    """
    모든 접두사/접미사를 분리하지 않고 결합합니다.

    .. versionadded:: 0.11.0
    """
