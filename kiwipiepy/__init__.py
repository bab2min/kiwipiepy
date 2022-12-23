"""
.. include:: ./documentation.md
"""
from _kiwipiepy import Token
from kiwipiepy._version import __version__
from kiwipiepy._wrap import Kiwi, Sentence, TypoTransformer, TypoDefinition, HSDataset
import kiwipiepy.utils as utils
from kiwipiepy.const import Match, Option

Kiwi.__module__ = 'kiwipiepy'
TypoTransformer.__module__ = 'kiwipiepy'
TypoDefinition.__module__ = 'kiwipiepy'
Sentence.__module__ = 'kiwipiepy'
HSDataset.__module__ = 'kiwipiepy'

basic_typos = TypoTransformer([
    TypoDefinition(["ㅐ", "ㅔ"], ["ㅐ", "ㅔ"], 1.),
    TypoDefinition(["ㅐ", "ㅔ"], ["ㅒ", "ㅖ"], 1.5),
    TypoDefinition(["ㅒ", "ㅖ"], ["ㅐ", "ㅔ"], 1.5),
    TypoDefinition(["ㅒ", "ㅖ"], ["ㅒ", "ㅖ"], 1.),
    TypoDefinition(["ㅚ", "ㅙ", "ㅞ"], ["ㅚ", "ㅙ", "ㅞ", "ㅐ", "ㅔ"], 1.),
    TypoDefinition(["ㅝ"], ["ㅗ", "ㅓ"], 1.),
    TypoDefinition(["ㅟ", "ㅢ"], ["ㅣ"], 1.),
    TypoDefinition(["위", "의"], ["이"], float("inf")),
    TypoDefinition(["위", "의"], ["이"], 1., "any"),
    TypoDefinition(["자", "쟈"], ["자", "쟈"], 1.),
    TypoDefinition(["재", "쟤"], ["재", "쟤"], 1.),
    TypoDefinition(["저", "져"], ["저", "져"], 1.),
    TypoDefinition(["제", "졔"], ["제", "졔"], 1.),
    TypoDefinition(["조", "죠", "줘"], ["조", "죠", "줘"], 1.),
    TypoDefinition(["주", "쥬"], ["주", "쥬"], 1.),
    TypoDefinition(["차", "챠"], ["차", "챠"], 1.),
    TypoDefinition(["채", "챼"], ["채", "챼"], 1.),
    TypoDefinition(["처", "쳐"], ["처", "쳐"], 1.),
    TypoDefinition(["체", "쳬"], ["체", "쳬"], 1.),
    TypoDefinition(["초", "쵸", "춰"], ["초", "쵸", "춰"], 1.),
    TypoDefinition(["추", "츄"], ["추", "츄"], 1.),
    TypoDefinition(["유", "류"], ["유", "류"], 1.),
    TypoDefinition(["므", "무"], ["므", "무"], 1.),
    TypoDefinition(["브", "부"], ["브", "부"], 1.),
    TypoDefinition(["프", "푸"], ["프", "푸"], 1.),
    TypoDefinition(["르", "루"], ["르", "루"], 1.),
    TypoDefinition(["러", "뤄"], ["러", "뤄"], 1.),
    TypoDefinition(["\ㄲ", "\ㄳ"], ["\ㄱ", "\ㄲ", "\ㄳ"], 1.5),
    TypoDefinition(["\ㄵ", "\ㄶ"], ["\ㄴ", "\ㄵ", "\ㄶ"], 1.5),
    TypoDefinition(["\ㄺ", "\ㄻ", "\ㄼ", "\ㄽ", "\ㄾ", "\ㄿ", "\ㅀ"], ["\ㄹ", "\ㄺ", "\ㄻ", "\ㄼ", "\ㄽ", "\ㄾ", "\ㄿ", "\ㅀ"], 1.5),
    TypoDefinition(["\ㅅ", "\ㅆ"], ["\ㅅ", "\ㅆ"], 1.),

    TypoDefinition(["안"], ["않"], 1.5),
    TypoDefinition(["맞추", "맞히"], ["맞추", "맞히"], 1.5),
    TypoDefinition(["맞춰", "맞혀"], ["맞춰", "맞혀"], 1.5),
    TypoDefinition(["받치", "바치", "받히"], ["받치", "바치", "받히"], 1.5),
    TypoDefinition(["받쳐", "바쳐", "받혀"], ["받쳐", "바쳐", "받혀"], 1.5),
    TypoDefinition(["던", "든"], ["던", "든"], 1.),
    TypoDefinition(["때", "데"], ["때", "데"], 1.5),
    TypoDefinition(["빛", "빚"], ["빛", "빚"], 1.),

    TypoDefinition(["\ㄷ이", "지"], ["\ㄷ이", "지"], 1.),
    TypoDefinition(["\ㄷ여", "져"], ["\ㄷ여", "져"], 1.),
    TypoDefinition(["\ㅌ이", "치"], ["\ㅌ이", "치"], 1.),
    TypoDefinition(["\ㅌ여", "쳐"], ["\ㅌ여", "쳐"], 1.),
	
    TypoDefinition(["ㄱ", "ㄲ"], ["ㄱ", "ㄲ"], 1., "applosive"),
    TypoDefinition(["ㄷ", "ㄸ"], ["ㄷ", "ㄸ"], 1., "applosive"),
    TypoDefinition(["ㅂ", "ㅃ"], ["ㅂ", "ㅃ"], 1., "applosive"),
    TypoDefinition(["ㅅ", "ㅆ"], ["ㅅ", "ㅆ"], 1., "applosive"),
    TypoDefinition(["ㅈ", "ㅉ"], ["ㅈ", "ㅉ"], 1., "applosive"),

    TypoDefinition(["\ㅎㅎ", "\ㄱㅎ", "\ㅎㄱ"], ["\ㅎㅎ", "\ㄱㅎ", "\ㅎㄱ"], 1.),

    TypoDefinition(["\ㄱㄴ", "\ㄲㄴ", "ᆪㄴ", "ᆿㄴ", "ᆼㄴ"], ["\ㄱㄴ", "\ㄲㄴ", "ᆪㄴ", "ᆿㄴ", "ᆼㄴ"], 1.),
    TypoDefinition(["\ㄱㅁ", "\ㄲㅁ", "ᆪㅁ", "ᆿㅁ", "ᆼㅁ"], ["\ㄱㅁ", "\ㄲㅁ", "ᆪㅁ", "ᆿㅁ", "ᆼㅁ"], 1.),
    TypoDefinition(["\ㄱㄹ", "\ㄲㄹ", "ᆪㄹ", "ᆿㄹ", "ᆼㄹ", "ᆼㄴ",], ["\ㄱㄹ", "\ㄲㄹ", "ᆪㄹ", "ᆿㄹ", "ᆼㄹ", "ᆼㄴ",], 1.),
    TypoDefinition(["\ㄷㄴ", "\ㅅㄴ", "\ㅆㄴ", "\ㅈㄴ", "ᆾㄴ", "ᇀㄴ", "\ㄴㄴ"], ["\ㄷㄴ", "\ㅅㄴ", "\ㅆㄴ", "\ㅈㄴ", "ᆾㄴ", "ᇀㄴ", "\ㄴㄴ"], 1.),
    TypoDefinition(["\ㄷㅁ", "\ㅅㅁ", "\ㅆㅁ", "\ㅈㅁ", "ᆾㅁ", "ᇀㅁ", "\ㄴㅁ"], ["\ㄷㅁ", "\ㅅㅁ", "\ㅆㅁ", "\ㅈㅁ", "ᆾㅁ", "ᇀㅁ", "\ㄴㅁ"], 1.),
    TypoDefinition(["\ㄷㄹ", "\ㅅㄹ", "\ㅆㄹ", "\ㅈㄹ", "ᆾㄹ", "ᇀㄹ", "\ㄴㄹ", "\ㄴㄴ",], ["\ㄷㄹ", "\ㅅㄹ", "\ㅆㄹ", "\ㅈㄹ", "ᆾㄹ", "ᇀㄹ", "\ㄴㄹ", "\ㄴㄴ",], 1.),
    TypoDefinition(["\ㅂㄴ", "ᆹㄴ", "ᇁㄴ", "\ㅁㄴ"], ["\ㅂㄴ", "ᆹㄴ", "ᇁㄴ", "\ㅁㄴ"], 1.),
    TypoDefinition(["\ㅂㅁ", "ᆹㅁ", "ᇁㅁ", "\ㅁㅁ"], ["\ㅂㅁ", "ᆹㅁ", "ᇁㅁ", "\ㅁㅁ"], 1.),
    TypoDefinition(["\ㅂㄹ", "ᆹㄹ", "ᇁㄹ", "\ㅁㄹ", "\ㅁㄴ",], ["\ㅂㄹ", "ᆹㄹ", "ᇁㄹ", "\ㅁㄹ", "\ㅁㄴ",], 1.),
    TypoDefinition(["\ㄴㄹ", "\ㄴㄴ", "\ㄹㄹ", "\ㄹㄴ"], ["\ㄴㄹ", "\ㄴㄴ", "\ㄹㄹ", "\ㄹㄴ"], 1.),
	
    TypoDefinition(["\ㄱㅇ", "ㄱ"], ["\ㄱㅇ", "ㄱ"], 1., "vowel"),
    TypoDefinition(["\ㄲㅇ", "ㄲ"], ["\ㄲㅇ", "ㄲ"], 1., "vowel"),
    TypoDefinition(["\ㄴㅇ", "\ㄴㅎ", "ㄴ"], ["\ㄴㅇ", "\ㄴㅎ", "ㄴ"], 1., "vowel"),
    TypoDefinition(["\ㄵㅇ", "\ㄴㅈ"], ["\ㄵㅇ", "\ㄴㅈ"], 1., "vowel"),
    TypoDefinition(["\ㄶㅇ", "ㄴ"], ["\ㄶㅇ", "ㄴ"], 1., "vowel"),
    TypoDefinition(["\ㄷㅇ", "ㄷ"], ["\ㄷㅇ", "ㄷ"], 1., "vowel"),
    TypoDefinition(["\ㄹㅇ", "\ㄹㅎ", "ㄹ"], ["\ㄹㅇ", "\ㄹㅎ", "ㄹ"], 1., "vowel"),
    TypoDefinition(["\ㄺㅇ", "\ㄹㄱ"], ["\ㄺㅇ", "\ㄹㄱ"], 1., "vowel"),
    TypoDefinition(["\ㄺㅎ", "\ㄹㅋ"], ["\ㄺㅎ", "\ㄹㅋ"], 1., "vowel"),
    TypoDefinition(["\ㅁㅇ", "ㅁ"], ["\ㅁㅇ", "ㅁ"], 1., "vowel"),
    TypoDefinition(["\ㅂㅇ", "ㅂ"], ["\ㅂㅇ", "ㅂ"], 1., "vowel"),
    TypoDefinition(["\ㅅㅇ", "ㅅ"], ["\ㅅㅇ", "ㅅ"], 1., "vowel"),
    TypoDefinition(["\ㅆㅇ", "\ㅅㅅ", "ㅆ"], ["\ㅆㅇ", "\ㅅㅅ", "ㅆ"], 1., "vowel"),
    TypoDefinition(["\ㅈㅇ", "ㅈ"], ["\ㅈㅇ", "ㅈ"], 1., "vowel"),
    TypoDefinition(["\ㅊㅇ", "\ㅊㅎ", "\ㅈㅎ", "ㅊ"], ["\ㅊㅇ", "\ㅊㅎ", "\ㅈㅎ", "ㅊ"], 1., "vowel"),
    TypoDefinition(["\ㅋㅇ", "\ㅋㅎ", "\ㄱㅎ", "ㅋ"], ["\ㅋㅇ", "\ㅋㅎ", "\ㄱㅎ", "ㅋ"], 1., "vowel"),
    TypoDefinition(["\ㅌㅇ", "\ㅌㅎ", "\ㄷㅎ", "ㅌ"], ["\ㅌㅇ", "\ㅌㅎ", "\ㄷㅎ", "ㅌ"], 1., "vowel"),
    TypoDefinition(["\ㅍㅇ", "\ㅍㅎ", "\ㅂㅎ", "ㅍ"], ["\ㅍㅇ", "\ㅍㅎ", "\ㅂㅎ", "ㅍ"], 1., "vowel"),

    TypoDefinition(["은", "는"], ["은", "는"], 2.),
    TypoDefinition(["을", "를"], ["을", "를"], 2.),

    TypoDefinition(["ㅣ워", "ㅣ어", "ㅕ"], ["ㅣ워", "ㅣ어", "ㅕ"], 1.5),
    #TypoDefinition(["ㅡ아", "ㅏ"], ["ㅡ아", "ㅏ"], 1.5),
    #TypoDefinition(["ㅡ어", "ㅓ"], ["ㅡ어", "ㅓ"], 1.5),
    #TypoDefinition(["ㅡ오", "ㅗ"], ["ㅡ오", "ㅗ"], 1.5),
    #TypoDefinition(["ㅡ우", "ㅜ"], ["ㅡ우", "ㅜ"], 1.5),
])
'''
내장된 기본 오타 정보입니다.
'''
