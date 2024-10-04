# Kiwipiepy, Python용 Kiwi 패키지

https://github.com/bab2min/kiwipiepy

[![PyPI version](https://badge.fury.io/py/kiwipiepy.svg)](https://pypi.org/project/kiwipiepy/)

Python3 API 문서: https://bab2min.github.io/kiwipiepy 

[Kiwi 0.5 버전](https://github.com/bab2min/kiwi)부터는 Python3용 API를 제공합니다. 이 프로젝트를 빌드하여 Python에 모듈을 import해서 사용하셔도 좋고, 
혹은 더 간편하게 pip를 이용하여 이미 빌드된 kiwipiepy 모듈을 설치하셔도 좋습니다.
```console
$ pip install --upgrade pip
$ pip install kiwipiepy
```
또는
```console
$ pip3 install --upgrade pip
$ pip3 install kiwipiepy
```

현재 kiwipiepy 패키지는 Vista 버전 이상의 Windows OS 및 Linux, macOS 10.12 이상을 지원합니다.

**macOS M1** 등 binary distribution이 제공되지 않는 환경에서는 설치시 소스 코드 컴파일을 위해 **cmake3.12 이상이 필요**합니다.
```console
$ pip install cmake
$ pip install --upgrade pip
$ pip install kiwipiepy
```

## 테스트해보기

Kiwi 0.6.3 버전부터는 설치 후 바로 테스트할 수 있도록 대화형 인터페이스를 지원합니다. pip를 통해 설치가 완료된 후 다음과 같이 실행하여 형태소 분석기를 시험해볼 수 있습니다.
```console
$ python -m kiwipiepy
```
또는
```console
$ python3 -m kiwipiepy
```
대화형 인터페이스가 시작되면, 원하는 문장을 입력해 바로 형태소 분석결과를 확인할 수 있습니다.
```python
>> 안녕?
[Token(form='안녕', tag='IC', start=0, len=2), Token(form='?', tag='SF', start=2, len=3)]
```
인터페이스를 종료하려면 Ctrl + C 를 누르십시오.

Kiwi에서 사용하는 품사 태그는 세종 말뭉치의 품사 태그를 기초로 하고 일부 태그들을 개량하여 사용하고 있습니다. 자세한 태그 체계에 대해서는 [여기](https://github.com/bab2min/kiwipiepy#%ED%92%88%EC%82%AC-%ED%83%9C%EA%B7%B8)를 참조하십시오.

## 간단 예제

```python
>>> from kiwipiepy import Kiwi
>>> kiwi = Kiwi()
# tokenize 함수로 형태소 분석 결과를 얻을 수 있습니다.
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

# 불용어 관리를 위한 Stopwords 클래스도 제공합니다.
>>> from kiwipiepy.utils import Stopwords
>>> stopwords = Stopwords()
>>> kiwi.tokenize("분석 결과에서 불용어만 제외하고 출력할 수도 있다.", stopwords=stopwords)
[Token(form='분석', tag='NNG', start=0, len=2),
 Token(form='결과', tag='NNG', start=3, len=2),
 Token(form='불', tag='XPN', start=8, len=1),
 Token(form='용어', tag='NNG', start=9, len=2),
 Token(form='제외', tag='NNG', start=13, len=2),
 Token(form='출력', tag='NNG', start=18, len=2)]

# add, remove 메소드를 이용해 불용어 목록에 단어를 추가하거나 삭제할 수도 있습니다.
>>> stopwords.add(('결과', 'NNG'))
>>> kiwi.tokenize("분석 결과에서 불용어만 제외하고 출력할 수도 있다.", stopwords=stopwords)
[Token(form='분석', tag='NNG', start=0, len=2),
 Token(form='불', tag='XPN', start=8, len=1),
 Token(form='용어', tag='NNG', start=9, len=2),
 Token(form='제외', tag='NNG', start=13, len=2),
 Token(form='출력', tag='NNG', start=18, len=2)]

>>> tokens = kiwi.tokenize("각 토큰은 여러 정보를 담고 있습니다.")
>>> tokens[0]
Token(form='각', tag='MM', start=0, len=1)
>>> tokens[0].form # 형태소의 형태 정보
'각'
>>> tokens[0].tag # 형태소의 품사 정보
'MM'
>>> tokens[0].start # 시작 및 끝 지점 (문자 단위)
0
>>> tokens[0].end
1
>>> tokens[0].word_position # 현 문장에서의 어절 번호
0
>>> tokens[0].sent_position # 형태소가 속한 문장 번호
0
>>> tokens[0].line_number # 형태소가 속한 줄의 번호
0

# 문장 분리 기능도 지원합니다.
>>> kiwi.split_into_sents("여러 문장으로 구성된 텍스트네 이걸 분리해줘")
[Sentence(text='여러 문장으로 구성된 텍스트네', start=0, end=16, tokens=None),
 Sentence(text='이걸 분리해줘', start=17, end=24, tokens=None)]

# 문장 분리와 형태소 분석을 함께 수행할 수도 있습니다.
>>> kiwi.split_into_sents("여러 문장으로 구성된 텍스트네 이걸 분리해줘", return_tokens=True)
[Sentence(text='여러 문장으로 구성된 텍스트네', start=0, end=16, tokens=[
  Token(form='여러', tag='MM', start=0, len=2), 
  Token(form='문장', tag='NNG', start=3, len=2), 
  Token(form='으로', tag='JKB', start=5, len=2), 
  Token(form='구성', tag='NNG', start=8, len=2), 
  Token(form='되', tag='XSV', start=10, len=1), 
  Token(form='ᆫ', tag='ETM', start=11, len=0), 
  Token(form='텍스트', tag='NNG', start=12, len=3), 
  Token(form='이', tag='VCP', start=15, len=1), 
  Token(form='네', tag='EF', start=15, len=1)]),
 Sentence(text='이걸 분리해줘', start=17, end=24, tokens=[
  Token(form='이거', tag='NP', start=17, len=2), 
  Token(form='ᆯ', tag='JKO', start=19, len=0), 
  Token(form='분리', tag='NNG', start=20, len=2), 
  Token(form='하', tag='XSV', start=22, len=1), 
  Token(form='어', tag='EC', start=22, len=1), 
  Token(form='주', tag='VX', start=23, len=1), 
  Token(form='어', tag='EF', start=23, len=1)])]

# 사전에 새로운 단어를 추가할 수 있습니다.
>>> kiwi.add_user_word("김갑갑", "NNP")
True
>>> kiwi.tokenize("김갑갑이 누구야")
[Token(form='김갑갑', tag='NNP', start=0, len=3),
 Token(form='이', tag='JKS', start=3, len=1),
 Token(form='누구', tag='NP', start=5, len=2),
 Token(form='야', tag='JKV', start=7, len=1)]

# v0.11.0 신기능
# 0.11.0 버전부터는 사용자 사전에 동사/형용사를 추가할 때, 그 활용형도 함께 등재됩니다.
# 사전에 등재되어 있지 않은 동사 `팅기다`를 분석하면, 엉뚱한 결과가 나옵니다.
>>> kiwi.tokenize('팅겼다')
[Token(form='팅기', tag='NNG', start=0, len=2),
 Token(form='하', tag='XSA', start=2, len=0), 
 Token(form='다', tag='EF', start=2, len=1)]

# 형태소 `팅기/VV`를 사전에 등록하면, 이 형태소의 모든 활용형이 자동으로 추가되기에
# `팅겼다`, `팅길` 등의 형태를 모두 분석해낼 수 있습니다.
>>> kiwi.add_user_word('팅기', 'VV')
True
>>> kiwi.tokenize('팅겼다')
[Token(form='팅기', tag='VV', start=0, len=2), 
 Token(form='었', tag='EP', start=1, len=1), 
 Token(form='다', tag='EF', start=2, len=1)]

# 또한 변형된 형태소를 일괄적으로 추가하여 대상 텍스트에 맞춰 분석 성능을 높일 수 있습니다.
>>> kiwi.tokenize("안녕하세영, 제 이름은 이세영이에영. 학생이세영?")
[Token(form='안녕', tag='NNG', start=0, len=2),
 Token(form='하', tag='XSA', start=2, len=1),
 Token(form='시', tag='EP', start=3, len=1),
 Token(form='어', tag='EC', start=3, len=1),
 Token(form='영', tag='MAG', start=4, len=1), # 오분석
 Token(form=',', tag='SP', start=5, len=1),
 Token(form='저', tag='NP', start=7, len=1),
 Token(form='의', tag='JKG', start=7, len=1),
 Token(form='이름', tag='NNG', start=9, len=2),
 Token(form='은', tag='JX', start=11, len=1),
 Token(form='이세영', tag='NNP', start=13, len=3),
 Token(form='이', tag='JKS', start=16, len=1),
 Token(form='에', tag='IC', start=17, len=1),
 Token(form='영', tag='NR', start=18, len=1),
 Token(form='.', tag='SF', start=19, len=1),
 Token(form='님', tag='NNG', start=21, len=1),
 Token(form='도', tag='JX', start=22, len=1),
 Token(form='학생', tag='NNG', start=24, len=2),
 Token(form='이세영', tag='NNP', start=26, len=3), # 오분석
 Token(form='?', tag='SF', start=29, len=1)]

# 종결어미(EF) 중 '요'로 끝나는 것들을 '영'으로 대체하여 일괄 삽입합니다. 
# 이 때 변형된 종결어미에는 -3의 페널티를 부여하여 원 형태소보다 우선하지 않도록 합니다.
# 새로 삽입된 형태소들이 반환됩니다.
>>> kiwi.add_re_rule('EF', '요$', '영', -3)
['어영', '에영', '지영', '잖아영', '거든영', 'ᆯ까영', '네영', '구영', '나영', '군영', ..., '으니깐영']

# 동일한 문장을 재분석하면 분석 결과가 개선된 것을 확인할 수 있습니다.
>>> kiwi.tokenize("안녕하세영, 제 이름은 이세영이에영. 님도 학생이세영?")
[Token(form='안녕', tag='NNG', start=0, len=2),
 Token(form='하', tag='XSA', start=2, len=1),
 Token(form='시', tag='EP', start=3, len=1),
 Token(form='어영', tag='EF', start=3, len=2), # 분석 결과 개선
 Token(form=',', tag='SP', start=5, len=1),
 Token(form='저', tag='NP', start=7, len=1),
 Token(form='의', tag='JKG', start=7, len=1),
 Token(form='이름', tag='NNG', start=9, len=2),
 Token(form='은', tag='JX', start=11, len=1),
 Token(form='이세영', tag='NNP', start=13, len=3),
 Token(form='이', tag='VCP', start=16, len=1),
 Token(form='에영', tag='EF', start=17, len=2),
 Token(form='.', tag='SF', start=19, len=1),
 Token(form='님', tag='NNG', start=21, len=1),
 Token(form='도', tag='JX', start=22, len=1),
 Token(form='학생', tag='NNG', start=24, len=2),
 Token(form='이', tag='VCP', start=26, len=1),
 Token(form='시', tag='EP', start=27, len=1),
 Token(form='어영', tag='EF', start=27, len=2), # 분석 결과 개선
 Token(form='?', tag='SF', start=29, len=1)]
 
# 기분석 형태를 등록하여 원하는 대로 분석되지 않는 문자열을 교정할 수도 있습니다.
# 다음 문장의 `사겼대`는 오타가 들어간 형태라 제대로 분석되지 않습니다.
>>> kiwi.tokenize('걔네 둘이 사겼대')
[Token(form='걔', tag='NP', start=0, len=1), 
 Token(form='네', tag='XSN', start=1, len=1), 
 Token(form='둘', tag='NR', start=3, len=1), 
 Token(form='이', tag='JKS', start=4, len=1), 
 Token(form='사', tag='NR', start=6, len=1), 
 Token(form='기', tag='VV', start=7, len=1), 
 Token(form='었', tag='EP', start=7, len=1), 
 Token(form='대', tag='EF', start=8, len=1)]
# 다음과 같이 add_pre_analyzed_word 메소드를 이용하여 이를 교정할 수 있습니다.
>>> kiwi.add_pre_analyzed_word('사겼대', ['사귀/VV', '었/EP', '대/EF'], -3)
True
# 그 뒤 동일한 문장을 다시 분석해보면 결과가 바뀐 것을 확인할 수 있습니다.
>>> kiwi.tokenize('걔네 둘이 사겼대')
[Token(form='걔', tag='NP', start=0, len=1), 
 Token(form='네', tag='XSN', start=1, len=1), 
 Token(form='둘', tag='NR', start=3, len=1), 
 Token(form='이', tag='JKS', start=4, len=1), 
 Token(form='사귀', tag='VV', start=6, len=3), 
 Token(form='었', tag='EP', start=6, len=3), 
 Token(form='대', tag='EF', start=6, len=3)]
# 단, 사귀/VV, 었/EP, 대/EF의 시작위치가 모두 6, 길이가 모두 3으로 잘못 잡히는 문제가 보입니다.
# 이를 고치기 위해서는 add_pre_analyzed_word 시 각 형태소의 위치정보도 함께 입력해주어야합니다.
>>> kiwi = Kiwi()
>>> kiwi.add_pre_analyzed_word('사겼대', [('사귀', 'VV', 0, 2), ('었', 'EP', 1, 2), ('대', 'EF', 2, 3)], -3)
True
>>> kiwi.tokenize('걔네 둘이 사겼대')
[Token(form='걔', tag='NP', start=0, len=1), 
 Token(form='네', tag='XSN', start=1, len=1), 
 Token(form='둘', tag='NR', start=3, len=1), 
 Token(form='이', tag='JKS', start=4, len=1), 
 Token(form='사귀', tag='VV', start=6, len=2, 
 Token(form='었', tag='EP', start=7 len=1, 
 Token(form='대', tag='EF', start=8 len=1]

# v0.12.0 신기능
# 0.12.0 버전부터는 형태소를 결합하여 문장으로 복원하는 기능이 추가되었습니다.
>>> kiwi.join([('길', 'NNG'), ('을', 'JKO'), ('묻', 'VV'), ('어요', 'EF')])
'길을 물어요'
>>> kiwi.join([('흙', 'NNG'), ('이', 'JKS'), ('묻', 'VV'), ('어요', 'EF')])
'흙이 묻어요'

# v0.13.0 신기능
# 더 강력한 언어 모델인 SkipBigram(sbg)이 추가되었습니다.
# 기존의 knlm과 달리 먼 거리에 있는 형태소를 고려할 수 있습니다.
>>> kiwi = Kiwi(model_type='knlm')
>>> kiwi.tokenize('이 번호로 전화를 이따가 꼭 반드시 걸어.')
[Token(form='이', tag='MM', start=0, len=1), 
 Token(form='번호', tag='NNG', start=2, len=2), 
 Token(form='로', tag='JKB', start=4, len=1), 
 Token(form='전화', tag='NNG', start=6, len=2), 
 Token(form='를', tag='JKO', start=8, len=1), 
 Token(form='이따가', tag='MAG', start=10, len=3), 
 Token(form='꼭', tag='MAG', start=14, len=1), 
 Token(form='반드시', tag='MAG', start=16, len=3), 
 Token(form='걷', tag='VV-I', start=20, len=1),  # 걷다/걸다 중 틀리게 '걷다'를 선택했음.
 Token(form='어', tag='EF', start=21, len=1), 
 Token(form='.', tag='SF', start=22, len=1)]

>>> kiwi = Kiwi(model_type='sbg')
>>> kiwi.tokenize('이 번호로 전화를 이따가 꼭 반드시 걸어.')
[Token(form='이', tag='MM', start=0, len=1), 
 Token(form='번호', tag='NNG', start=2, len=2), 
 Token(form='로', tag='JKB', start=4, len=1), 
 Token(form='전화', tag='NNG', start=6, len=2), 
 Token(form='를', tag='JKO', start=8, len=1), 
 Token(form='이따가', tag='MAG', start=10, len=3), 
 Token(form='꼭', tag='MAG', start=14, len=1), 
 Token(form='반드시', tag='MAG', start=16, len=3), 
 Token(form='걸', tag='VV', start=20, len=1), # 걷다/걸다 중 바르게 '걸다'를 선택했음.
 Token(form='어', tag='EC', start=21, len=1), 
 Token(form='.', tag='SF', start=22, len=1)]

# 또한 오타 교정 기능이 추가되었습니다.
# 간단한 오타를 교정하여, 사소한 오타 때문에 전체 분석 결과가 어긋나는 문제를 해결할 수 있습니다.
>>> kiwi = Kiwi(model_type='sbg', typos='basic')
>>> kiwi.tokenize('외않됀대?') # 오타 교정 사용 시 로딩 시간이 5~10초 정도 소요됨
[Token(form='왜', tag='MAG', start=0, len=1),
 Token(form='안', tag='MAG', start=1, len=1),
 Token(form='되', tag='VV', start=2, len=1),
 Token(form='ᆫ대', tag='EF', start=2, len=2),
 Token(form='?', tag='SF', start=4, len=1)]

>>> kiwi.tokenize('장례희망이 뭐냐는 선섕님의 질문에 벙어리가 됫따') 
[Token(form='장래', tag='NNG', start=0, len=2),
 Token(form='희망', tag='NNG', start=2, len=2), 
 Token(form='이', tag='JKS', start=4, len=1), 
 Token(form='뭐', tag='NP', start=6, len=1), 
 Token(form='이', tag='VCP', start=7, len=0), 
 Token(form='냐는', tag='ETM', start=7, len=2), 
 Token(form='선생', tag='NNG', start=10, len=2), 
 Token(form='님', tag='XSN', start=12, len=1), 
 Token(form='의', tag='JKG', start=13, len=1), 
 Token(form='질문', tag='NNG', start=15, len=2), 
 Token(form='에', tag='JKB', start=17, len=1), 
 Token(form='벙어리', tag='NNG', start=19, len=3), 
 Token(form='가', tag='JKC', start=22, len=1), 
 Token(form='되', tag='VV', start=24, len=1), 
 Token(form='엇', tag='EP', start=24, len=1), 
 Token(form='다', tag='EF', start=25, len=1)]

# 0.17.1에서는 연철에 대한 오타 교정이 추가되었습니다.
# 받침 + 초성 ㅇ/ㅎ 꼴을 잘못 이어적은 경우에 대해 교정이 가능합니다.
>>> kiwi = Kiwi(typos='continual')
>>> kiwi.tokenize('오늘사무시레서')
[Token(form='오늘', tag='NNG', start=0, len=2),
 Token(form='사무실', tag='NNG', start=2, len=4),
 Token(form='에서', tag='JKB', start=5, len=2)]
>>> kiwi.tokenize('지가캤어요')
[Token(form='지각', tag='NNG', start=0, len=3),
 Token(form='하', tag='XSV', start=2, len=1),
 Token(form='었', tag='EP', start=2, len=1),
 Token(form='어요', tag='EF', start=3, len=2)]

# 기본 오타 교정에 연철 오타 교정까지 함께 사용할 수도 있습니다.
>>> kiwi = Kiwi(typos='basic_with_continual')
>>> kiwi.tokenize('웨 지가캤니?')
[Token(form='왜', tag='MAG', start=0, len=1),
 Token(form='지각', tag='NNG', start=2, len=3),
 Token(form='하', tag='XSV', start=4, len=1),
 Token(form='었', tag='EP', start=4, len=1),
 Token(form='니', tag='EC', start=5, len=1),
 Token(form='?', tag='SF', start=6, len=1)]

# 0.19.0 버전에서는 장음화 오류(한 음절을 여러 음절로 늘려 적는 오류)가 
# 포함된 텍스트를 교정하는 기능도 추가되었습니다.
>>> kiwi = Kiwi(typos='lengthening')
>>> kiwi.tokenize('지이인짜 귀여워요')
[Token(form='진짜', tag='MAG', start=0, len=4), 
 Token(form='귀엽', tag='VA-I', start=5, len=3), 
 Token(form='어요', tag='EF', start=7, len=2)]

# 기본 오타 교정 + 연철 오타 교정 + 장음화 오류 교정을 함께 사용할 수도 있습니다.
>>> kiwi = Kiwi(typos='basic_with_continual_and_lengthening')
>>> kiwi.tokenize('지이인짜 기여워요~ 마니 좋아해')
[Token(form='진짜', tag='MAG', start=0, len=4),
 Token(form='귀엽', tag='VA-I', start=5, len=3),
 Token(form='어요', tag='EF', start=7, len=2), 
 Token(form='~', tag='SO', start=9, len=1), 
 Token(form='많이', tag='MAG', start=11, len=2), 
 Token(form='좋아하', tag='VV', start=14, len=3), 
 Token(form='어', tag='EF', start=16, len=1)]

# 0.17.0 버전부터는 사용자 사전에 공백이 있는 단어를 추가할 수 있습니다.
>>> kiwi = Kiwi()
# '대학생 선교회'라는 단어를 등록합니다.
>>> kiwi.add_user_word('대학생 선교회', 'NNP')
True

# 등록한 것과 동일한 형태에서는
# 당연히 잘 분석됩니다.
>>> kiwi.tokenize('대학생 선교회에서') 
[Token(form='대학생 선교회', tag='NNP', start=0, len=7),
 Token(form='에서', tag='JKB', start=7, len=2)]

# 추가로 공백이 없는 형태에도 일치가 가능합니다.
>>> kiwi.tokenize('대학생선교회에서') 
kiwi.tokenize('대학생선교회에서')  
[Token(form='대학생 선교회', tag='NNP', start=0, len=6),
 Token(form='에서', tag='JKB', start=6, len=2)]

# 탭 문자나 줄바꿈 문자 등이 들어가도 일치가 가능합니다.
# 연속한 공백 문자는 공백 1번과 동일하게 처리합니다.
>>> kiwi.tokenize('대학생 \t \n 선교회에서') 
[Token(form='대학생 선교회', tag='NNP', start=0, len=11),
 Token(form='에서', tag='JKB', start=11, len=2)]

# 그러나 사전 등재 시 공백이 없던 지점에
# 공백이 있는 경우에는 일치가 불가능합니다.
>>> kiwi.tokenize('대학 생선 교회에서')      
[Token(form='대학', tag='NNG', start=0, len=2),
 Token(form='생선', tag='NNG', start=3, len=2),
 Token(form='교회', tag='NNG', start=6, len=2),
 Token(form='에서', tag='JKB', start=8, len=2)]

# space_tolerance를 2로 설정하여
# 공백이 두 개까지 틀린 경우를 허용하도록 하면
# '대학 생선 교회'에도 '대학생 선교회'가 일치하게 됩니다.
>>> kiwi.space_tolerance = 2
>>> kiwi.tokenize('대학 생선 교회에서')
[Token(form='대학생 선교회', tag='NNP', start=0, len=8),
 Token(form='에서', tag='JKB', start=8, len=2)]

# 0.18.0 버전에서는 외국어 문자, 이모지에 대한 지원이 강화되었습니다.
# 화면에 표시되는 글자 단위로 토큰이 분할됩니다.
>>> kiwi.tokenize('😂☝🏻☝🏿')
[Token(form='😂', tag='W_EMOJI', start=0, len=1), 
 Token(form='☝🏻', tag='W_EMOJI', start=1, len=2), 
 Token(form='☝🏿', tag='W_EMOJI', start=3, len=2)]
# 참고: v0.17의 결과
# [Token(form='😂☝🏻☝🏿', tag='SW', start=0, len=5)]

# script 필드가 추가되어 해당 문자가
# 유니코드 상에서 어떤 영역에 속하는지 확인할 수 있습니다.
# SW, SH, SL, W_EMOJI 태그에 대해서만 script값이 부여됩니다.
>>> tokens = kiwi.tokenize('ひらがなカタカナ')
>>> tokens
[Token(form='ひらがなカタカナ', tag='SW', start=0, len=8)]
>>> tokens[0].script
'Kana'

>>> tokens = kiwi.tokenize('résumé')
>>> tokens
[Token(form='résumé', tag='SL', start=0, len=6)]
# 참고 v0.17까지의 결과
# [Token(form='r', tag='SL', start=0, len=1), 
#  Token(form='é', tag='SW', start=1, len=1), 
#  Token(form='sum', tag='SL', start=2, len=3), 
#  Token(form='é', tag='SW', start=5, len=1)]
>>> tokens[0].script
'Latin'

>>> tokens = kiwi.tokenize('ἥρως')
>>> tokens
[Token(form='ἥρως', tag='SW', start=0, len=4)]
>>> tokens[0].script
'Greek and Coptic'

>>> tokens = kiwi.tokenize('ฉันชอบกินข้าวผัด')
>>> tokens
[Token(form='ฉันชอบกินข้าวผัด', tag='SW', start=0, len=16)]
>>> tokens[0].script
'Thai'

# 0.18.1버전부터는 받침만으로 구성된 형태소 출력시 
# 호환용 자모를 사용하는 옵션을 제공합니다.
>>> kiwi.tokenize('예쁜데')
[Token(form='예쁘', tag='VA', start=0, len=2),
 Token(form='ᆫ데', tag='EF', start=1, len=2)]
>>> kiwi.tokenize('예쁜데', compatible_jamo=True) 
[Token(form='예쁘', tag='VA', start=0, len=2),
 Token(form='ㄴ데', tag='EF', start=1, len=2)] 
# 받침 ᆫ이 호환용 자모인 ㄴ으로 변환되어 출력됨
```

## 시작하기

kiwipiepy 패키지 설치가 성공적으로 완료되었다면, 다음과 같이 패키지를 import후 Kiwi 객체를 생성했을때 오류가 발생하지 않습니다.
```python
from kiwipiepy import Kiwi, Match
kiwi = Kiwi()
```
Kiwi 생성자는 다음과 같습니다.
```python
Kiwi(num_workers=0, model_path=None, load_default_dict=True, integrate_allomorph=True, model_type='knlm', typos=None, typo_cost_threshold=2.5)
```
* `num_workers`:  2 이상이면 단어 추출 및 형태소 분석에 멀티 코어를 활용하여 조금 더 빠른 속도로 분석을 진행할 수 있습니다. <br>
1인 경우 단일 코어만 활용합니다. num_workers가 0이면 현재 환경에서 사용가능한 모든 코어를 활용합니다. <br>
생략 시 기본값은 0입니다.
* `model_path`: 형태소 분석 모델이 있는 경로를 지정합니다. 생략시 `kiwipiepy_model` 패키지로부터 모델 경로를 불러옵니다.
* `load_default_dict`: 추가 사전을 로드합니다. 추가 사전은 위키백과의 표제어 타이틀로 구성되어 있습니다. 이 경우 로딩 및 분석 시간이 약간 증가하지만 다양한 고유명사를 좀 더 잘 잡아낼 수 있습니다. 분석 결과에 원치 않는 고유명사가 잡히는 것을 방지하려면 이를 False로 설정하십시오.
* `integrate_allomorph`: 어미 중, '아/어', '았/었'과 같이 동일하지만 음운 환경에 따라 형태가 달라지는 이형태들을 자동으로 통합합니다.
* `model_type`: 형태소 분석에 사용할 언어 모델을 지정합니다. `'knlm'`, `'sbg'` 중 하나를 선택할 수 있습니다. `'sbg'` 는 상대적으로 느리지만 먼 거리에 있는 형태소 간의 관계를 포착할 수 있습니다.
* `typos`: 형태소 분석 시 간단한 오타를 교정합니다. `None`으로 설정 시 교정을 수행하지 않습니다.
* `typo_cost_threshold`: 오타 교정을 허용할 최대 오타 비용을 설정합니다.

kiwi 객체는 크게 다음 세 종류의 작업을 수행할 수 있습니다.
* 코퍼스로부터 미등록 단어 추출
* 사용자 사전 관리
* 형태소 분석

### 코퍼스로부터 미등록 단어 추출

Kiwi 0.5부터 새로 추가된 기능입니다. 자주 등장하는 문자열의 패턴을 파악하여 단어로 추정되는 문자열을 추출해줍니다. 
이 기능의 기초적인 아이디어는 https://github.com/lovit/soynlp 의 Word Extraction 기법을 바탕으로 하고 있으며, 
이에 문자열 기반의 명사 확률을 조합하여 명사일 것으로 예측되는 단어만 추출합니다.

Kiwi가 제공하는 미등록 단어 추출 관련 메소드는 다음 두 가지입니다.
```python
Kiwi.extract_words(texts, min_cnt, max_word_len, min_score)
Kiwi.extract_add_words(texts, min_cnt, max_word_len, min_score, pos_score)
```
<details>
<summary><code>extract_words(texts, min_cnt=10, max_word_len=10, min_score=0.25, pos_score=-3.0, lm_filter=True)</code></summary>

* `texts`: 분석할 텍스트를 `Iterable[str]` 형태로 넣어줍니다. 자세한 건 아래의 예제를 참조해주세요.
* `min_cnt`: 추출할 단어가 입력 텍스트 내에서 최소 몇 번 이상 등장하는 지를 결정합니다. 입력 텍스트가 클 수록 그 값을 높여주시는게 좋습니다.
* `max_word_len`: 추출할 단어의 최대 길이입니다. 이 값을 너무 크게 설정할 경우 단어를 스캔하는 시간이 길어지므로 적절하게 조절해주시는 게 좋습니다.
* `min_score`:  추출할 단어의 최소 단어 점수입니다. 이 값을 낮출수록 단어가 아닌 형태가 추출될 가능성이 높아지고, 
반대로 이 값을 높일 수록 추출되는 단어의 개수가 줄어들므로 적절한 수치로 설정하실 필요가 있습니다. 기본값은 0.25입니다.
* `pos_score`: 추출할 단어의 최소 명사 점수입니다. 이 값을 낮출수록 명사가 아닌 단어들이 추출될 가능성이 높으며, 반대로 높일수록 추출되는 명사의 개수가 줄어듭니다. 기본값은 -3입니다.
* `lm_filter`: 품사 및 언어 모델을 이용한 필터링을 사용할 지 결정합니다.
```python
# 입력으로 str의 list를 줄 경우
inputs = list(open('test.txt', encoding='utf-8'))
kiwi.extract_words(inputs, min_cnt=10, max_word_len=10, min_score=0.25)

'''
위의 코드에서는 모든 입력을 미리 list로 저장해두므로
test.txt 파일이 클 경우 많은 메모리를 소모할 수 있습니다.

그 대신 파일에서 필요한 부분만 가져와 사용하려면(streaming)
아래와 같이 사용해야 합니다.
'''

class IterableTextFile:
    def __init__(self, path):
        self.path = path

    def __iter__(self):
        yield from open(path, encoding='utf-8')

kiwi.extract_words(IterableTextFile('test.txt'), min_cnt=10, max_word_len=10, min_score=0.25)
```
</details>
<hr>
<details>
<summary><code>extract_add_words(texts, min_cnt=10, max_word_len=10, min_score=0.25, pos_score=-3, lm_filter=True)</code></summary>

`extract_words` 와 동일하게 명사인 단어만 추출해줍니다. 
다만 이 메소드는 추출된 명사 후보를 자동으로 사용자 사전에 `NNP`로 등록하여 형태소 분석에 사용할 수 있게 해줍니다. 만약 이 메소드를 사용하지 않는다면 add_user_word 메소드를 사용하여 추출된 미등록 단어를 직접 사용자 사전에 등록해야 합니다.
</details>
<hr>

### 사용자 사전 관리

기존의 사전에 등록되지 않은 단어를 제대로 분석하기 위해서는 사용자 사전에 해당 단어를 등록해주어야 합니다. 
이는 extract_add_words를 통해서 자동으로 이뤄질 수도 있고, 수작업으로 직접 추가될 수도 있습니다. 
다음 메소드들은 사용자 사전을 관리하는데 사용되는 메소드들입니다.
```python
Kiwi.add_user_word(word, tag, score, orig_word=None)
Kiwi.add_pre_analyzed_word(form, analyzed, score)
Kiwi.add_rule(tag, replacer, score)
Kiwi.add_re_rule(tag, pattern, repl, score)
Kiwi.load_user_dictionary(user_dict_path)
```
<details>
<summary><code>add_user_word(word, tag='NNP', score=0.0, orig_word=None)</code></summary>

사용자 사전에 새 형태소를 등록합니다. 

* `word`: 등록할 형태소의 형태입니다. 현재는 띄어쓰기(공백문자)가 포함되지 않는 문자열만 단어로 등록할 수 있습니다.
* `tag`: 등록할 형태소의 품사입니다. 기본값은 NNP(고유명사)입니다.
* `score`: 등록할 형태소의 점수입니다. 
    동일한 형태라도 여러 경우로 분석될 가능성이 있는 경우에, 이 값이 클수록 해당 형태소가 더 우선권을 가지게 됩니다.
* `orig_word`: 추가할 형태소가 특정 형태소의 변이형인 경우 이 인자로 원본 형태소를 넘겨줄 수 있습니다. 없는 경우 생략할 수 있습니다. 
     이 값을 준 경우, 현재 사전 내에 `orig_word`/`tag` 조합의 형태소가 반드시 존재해야 하며, 그렇지 않으면 `ValueError` 예외를 발생시킵니다. 
     원본 형태소가 존재하는 경우 `orig_word`를 명시하면 더 정확한 분석 결과를 낼 수 있습니다.

형태소 삽입이 성공하면 `True`를, 동일한 형태소가 이미 존재하여 실패하면 `False`를 반환합니다.
</details>
<hr>

<details>
<summary><code>add_pre_analyzed_word(form, analyzed, score=0.0)</code></summary>

사용자 사전에 기분석 형태를 등록합니다. 이를 통해 특정 형태가 사용자가 원하는 형태로 형태소 분석이 되도록 유도할 수 있습니다.

* `form`: 기분석의 형태입니다.
* `analyzed`: `form`의 형태소 분석 결과.
    이 값은 (형태, 품사) 모양의 tuple, 혹은 (형태, 품사, 시작지점, 끝지점) 모양의 tuple로 구성된 Iterable이어야합니다.
    이 값으로 지정되는 형태소는 현재 사전 내에 반드시 존재해야 하며, 그렇지 않으면 `ValueError` 예외를 발생시킵니다.
* `score`: 추가할 형태소열의 가중치 점수. 
    해당 형태에 부합하는 형태소 조합이 여러 개가 있는 경우, 이 점수가 높을 단어가 더 우선권을 가집니다.

삽입이 성공하면 `True`를, 동일한 형태가 이미 존재하여 실패하면 `False`를 반환합니다.

이 메소드는 불규칙적인 분석 결과를 분석기에 추가하는 데에 용이합니다. 
예를 들어 `사귀다` 동사의 과거형은 `사귀었다`가 맞지만, 흔히 `사겼다`로 잘못 쓰이기도 합니다.
`사겼다`가 `사귀/VV + 었/EP + 다/EF`로 바르게 분석되도록 하는데에 이 메소드를 사용할 수 있습니다.

 ```python
kiwi.add_pre_analyzed_word('사겼다', ['사귀/VV', '었/EP', '다/EF'], -3)`
kiwi.add_pre_analyzed_word('사겼다', [('사귀', 'VV', 0, 2), ('었', 'EP', 1, 2), ('다', 'EF', 2, 3)], -3)
 ```

후자의 경우 분석 결과의 각 형태소가 원본 문자열에서 차지하는 위치를 정확하게 지정해줌으로써, 
Kiwi 분석 결과에서 해당 형태소의 `start`, `end`, `length`가 정확하게 나오도록 합니다.
</details>
<hr>

<details>
<summary><code>add_rule(tag, replacer, score)</code></summary>

규칙에 의해 변형된 형태소를 일괄적으로 추가합니다.
* `tag`: 추가할 형태소들의 품사
* `replacer`: 형태소를 변형시킬 규칙. 
    이 값은 호출가능한 Callable 형태로 제공되어야 하며, 원본 형태소 str를 입력으로 받아 변형된 형태소의 str을 반환해야합니다.
    만약 입력과 동일한 값을 반환하면 해당 변형 결과는 무시됩니다.
* `score`: 추가할 변형된 형태소의 가중치 점수. 
    해당 형태에 부합하는 형태소 조합이 여러 개가 있는 경우, 이 점수가 높을 단어가 더 우선권을 가집니다.

`replacer`에 의해 새로 생성된 형태소의 `list`를 반환합니다.
</details>
<hr>

<details>
<summary><code>add_re_rule(tag, pattern, repl, score)</code></summary>

`add_rule`메소드와 동일한 역할을 수행하되, 변형 규칙에 정규표현식을 사용합니다.

* `tag`: 추가할 형태소들의 품사
* `pattern`: 변형시킬 형태소의 규칙. 이 값은 `re.compile`로 컴파일가능한 정규표현식이어야 합니다.
* `repl`: `pattern`에 의해 발견된 패턴은 이 값으로 교체됩니다. Python3 정규표현식 모듈 내의 `re.sub` 함수의 `repl` 인자와 동일합니다.
* `score`: 추가할 변형된 형태소의 가중치 점수. 
    해당 형태에 부합하는 형태소 조합이 여러 개가 있는 경우, 이 점수가 높을 단어가 더 우선권을 가집니다.

`pattern`과 `repl`에 의해 새로 생성된 형태소의 `list`를 반환합니다.

이 메소드는 규칙에 의해 변형되는 이형태들을 일괄적으로 추가할 때 굉장히 용이합니다.
예를 들어 `-요`가 `-염`으로 교체된 종결어미들(`먹어염`, `뛰었구염`, `배불러염` 등)을 일괄 등록하기 위해서는
다음을 수행하면 됩니다.

```python
kiwi.add_re_rule('EF', r'요$', r'염', -3.0)
```

이런 이형태들을 대량으로 등록할 경우 이형태가 원본 형태보다 분석결과에서 높은 우선권을 가지지 않도록
score를 `-3` 이하의 값으로 설정하는걸 권장합니다.
</details>
<hr>

<details>
<summary><code>load_user_dictionary(user_dict_path)</code></summary>

파일로부터 사용자 사전을 읽어들입니다. 사용자 사전 파일은 UTF-8로 인코딩되어 있어야하며, 다음과 같은 형태로 구성되어야 합니다.
탭 문자(\t)로 각각의 필드는 분리되어야 하며, 단어 점수는 생략 가능합니다. 
```text
#으로 시작하는 줄은 주석 처리됩니다.
# 각 필드는 Tab(\t)문자로 구분됩니다.
#
# <단일 형태소를 추가하는 경우>
# (형태) \t (품사태그) \t (점수)
# * (점수)는 생략시 0으로 처리됩니다.
키위	NNP	-5.0
#
# <이미 존재하는 형태소의 이형태를 추가하는 경우>
# (이형태) \t (원형태소/품사태그) \t (점수)
# * (점수)는 생략시 0으로 처리됩니다.
기위	키위/NNG	-3.0
#
# <기분석 형태를 추가하는 경우>
# (형태) \t (원형태소/품사태그 + 원형태소/품사태그 + ...) \t (점수)
# * (점수)는 생략시 0으로 처리됩니다.
사겼다	사귀/VV + 었/EP + 다/EF	-1.0
#
# 현재는 공백을 포함하는 다어절 형태를 등록할 수 없습니다.
```
사전 파일을 성공적으로 읽어들이면, 사전을 통해 새로 추가된 형태소의 개수를 반환합니다. 

실제 예시에 대해서는 [Kiwi에 내장된 기본 사전 파일](https://raw.githubusercontent.com/bab2min/Kiwi/main/models/base/default.dict)을 참조해주세요.
</details>
<hr>

### 분석

kiwi을 생성하고, 사용자 사전에 단어를 추가하는 작업이 완료되었으면 다음 메소드를 사용하여 
형태소 분석, 문장 분리, 띄어쓰기 교정, 문장 복원 등의 작업을 수행할 수 있습니다.

```python
Kiwi.tokenize(text, match_option, normalize_coda=False, z_coda=True, split_complex=False, compatible_jamo=False, blocklist=None)
Kiwi.analyze(text, top_n, match_option, normalize_coda=False, z_coda=True, split_complex=False, compatible_jamo=False, blocklist=None)
Kiwi.split_into_sents(text, match_options=Match.ALL, normalize_coda=False, z_coda=True, split_complex=False, compatible_jamo=False, blocklist=None, return_tokens=False)
Kiwi.glue(text_chunks, insert_new_lines=None, return_space_insertions=False)
Kiwi.space(text, reset_whitespace=False)
Kiwi.join(morphs, lm_search=True)
Kiwi.template(format_str, cache=True)
``` 

<details>
<summary><code>tokenize(text, match_option=Match.ALL, normalize_coda=False, z_coda=True, split_complex=False, compatible_jamo=False, blocklist=None)</code></summary>
 
입력된 `text`를 형태소 분석하여 그 결과를 간단하게 반환합니다. 분석결과는 다음과 같이 `Token`의 리스트 형태로 반환됩니다.

```python
>> kiwi.tokenize('테스트입니다.')
[Token(form='테스트', tag='NNG', start=0, len=3), Token(form='이', tag='VCP', start=3, len=1), Token(form='ᆸ니다', tag='EF', start=4, len=2)]
```

`normalize_coda`는 ㅋㅋㅋ,ㅎㅎㅎ와 같은 초성체가 뒤따라와서 받침으로 들어갔을때 분석에 실패하는 문제를 해결해줍니다.
```python
>> kiwi.tokenize("안 먹었엌ㅋㅋ", normalize_coda=False)
[Token(form='안', tag='NNP', start=0, len=1), 
 Token(form='먹었엌', tag='NNP', start=2, len=3), 
 Token(form='ㅋㅋ', tag='SW', start=5, len=2)]
>> kiwi.tokenize("안 먹었엌ㅋㅋ", normalize_coda=True)
[Token(form='안', tag='MAG', start=0, len=1), 
 Token(form='먹', tag='VV', start=2, len=1), 
 Token(form='었', tag='EP', start=3, len=1), 
 Token(form='어', tag='EF', start=4, len=1), 
 Token(form='ㅋㅋㅋ', tag='SW', start=5, len=2)]
```
</details>
<hr>

<details>
<summary><code>analyze(text, top_n=1, match_option=Match.ALL, normalize_coda=False, z_coda=True, split_complex=False, compatible_jamo=False, blocklist=None)</code></summary>
 
입력된 `text`를 형태소 분석하여 그 결과를 반환합니다. 총 top_n개의 결과를 자세하게 출력합니다. 반환값은 다음과 같이 구성됩니다.
 
```python
[(분석결과1, 점수), (분석결과2, 점수), ... ]
```
 
분석결과는 다음과 같이 `Token`의 리스트 형태로 반환됩니다.

실제 예시는 다음과 같습니다.
```python
>> kiwi.analyze('테스트입니다.', top_n=5)
[([Token(form='테스트', tag='NNG', start=0, len=3), Token(form='이', tag='VCP', start=3, len=1), Token(form='ᆸ니다', tag='EF', start=4, len=2)], -25.217018127441406), 
 ([Token(form='테스트입니', tag='NNG', start=0, len=5), Token(form='다', tag='EC', start=5, len=1)], -40.741905212402344), 
 ([Token(form='테스트입니', tag='NNG', start=0, len=5), Token(form='다', tag='MAG', start=5, len=1)], -41.81024932861328), 
 ([Token(form='테스트입니', tag='NNG', start=0, len=5), Token(form='다', tag='EF', start=5, len=1)], -42.300254821777344), 
 ([Token(form='테스트', tag='NNG', start=0, len=3), Token(form='입', tag='NNG', start=3, len=1), Token(form='니다', tag='EF', start=4, len=2)], -45.86524200439453)
]
```
만약 text가 str의 iterable인 경우 여러 개의 입력을 병렬로 처리합니다. 이때의 반환값은 단일 text를 입력한 경우의 반환값의 iterable입니다.
Kiwi() 생성시 인자로 준 num_workers에 따라 여러 개의 스레드에서 작업이 동시에 처리됩니다. 반환되는 값은 입력되는 값의 순서와 동일합니다.
```python
>> result_iter = kiwi.analyze(['테스트입니다.', '테스트가 아닙니다.', '사실 맞습니다.'])
>> next(result_iter)
[([Token(form='테스트', tag='NNG', start=0, len=3), Token(form='이', tag='VCP', start=3, len=1), Token(form='ᆸ니다', tag='EF', start=4, len=2), Token(form='.', tag='SF', start=6, len=1)], -20.441545486450195)]
>> next(result_iter)
[([Token(form='테스트', tag='NNG', start=0, len=3), Token(form='가', tag='JKC', start=3, len=1), Token(form='아니', tag='VCN', start=5, len=2), Token(form='ᆸ니다', tag='EF', start=7, len=2), Token(form='.', tag='SF', start=9, len=1)], -30.23870277404785)]
>> next(result_iter)
[([Token(form='사실', tag='MAG', start=0, len=2), Token(form='맞', tag='VV', start=3, len=1), Token(form='습니다', tag='EF', start=4, len=3), Token(form='.', tag='SF', start=7, len=1)], -22.232769012451172)]
>> next(result_iter)
Traceback (most recent call last):
    File "<stdin>", line 1, in <module>
StopIteration
```
for 반복문을 사용하면 좀더 간단하고 편리하게 병렬 처리를 수행할 수 있습니다. 이는 대량의 텍스트 데이터를 분석할 때 유용합니다.
```python
>> for result in kiwi.analyze(long_list_of_text):
      tokens, score = result[0]
      print(tokens)
```
text를 str의 iterable로 준 경우 이 iterable을 읽어들이는 시점은 analyze 호출 이후일 수도 있습니다. 
따라서 이 인자가 다른 IO 자원(파일 입출력 등)과 연동되어 있다면 모든 분석이 끝나기 전까지 해당 자원을 종료하면 안됩니다.
```python
>> file = open('long_text.txt', encoding='utf-8')
>> result_iter = kiwi.analyze(file)
>> file.close() # 파일이 종료됨
>> next(result_iter) # 종료된 파일에서 분석해야할 다음 텍스트를 읽어들이려고 시도함
ValueError: I/O operation on closed file.
The above exception was the direct cause of the following exception:
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
SystemError: <built-in function next> returned a result with an error set
```
</details>
<hr>

<details>
<summary><code>split_into_sents( 
    text, 
    match_options=Match.ALL, 
    normalize_coda=False, 
    z_coda=True, 
    split_complex=False, 
    compatible_jamo=False,
    return_tokens=False
)</code></summary>
입력 텍스트를 문장 단위로 분할하여 반환합니다. 
이 메소드는 문장 분할 과정에서 내부적으로 형태소 분석을 사용하므로 문장 분할과 동시에 형태소 분석 결과를 얻는 데 사용할 수도 있습니다. `return_tokens`를 `True`로 설정하면 문장 분리와 함께 형태소 분석 결과도 출력합니다.

```python
>> kiwi.split_into_sents("여러 문장으로 구성된 텍스트네 이걸 분리해줘")
[Sentence(text='여러 문장으로 구성된 텍스트네', start=0, end=16, tokens=None),
 Sentence(text='이걸 분리해줘', start=17, end=24, tokens=None)]
>> kiwi.split_into_sents("여러 문장으로 구성된 텍스트네 이걸 분리해줘", return_tokens=True)
[Sentence(text='여러 문장으로 구성된 텍스트네', start=0, end=16, tokens=[
  Token(form='여러', tag='MM', start=0, len=2), 
  Token(form='문장', tag='NNG', start=3, len=2), 
  Token(form='으로', tag='JKB', start=5, len=2), 
  Token(form='구성', tag='NNG', start=8, len=2), 
  Token(form='되', tag='XSV', start=10, len=1), 
  Token(form='ᆫ', tag='ETM', start=11, len=0), 
  Token(form='텍스트', tag='NNG', start=12, len=3), 
  Token(form='이', tag='VCP', start=15, len=1), 
  Token(form='네', tag='EF', start=15, len=1)
 ]),
 Sentence(text='이걸 분리해줘', start=17, end=24, tokens=[
  Token(form='이거', tag='NP', start=17, len=2), 
  Token(form='ᆯ', tag='JKO', start=19, len=0), 
  Token(form='분리', tag='NNG', start=20, len=2), 
  Token(form='하', tag='XSV', start=22, len=1), 
  Token(form='어', tag='EC', start=22, len=1), 
  Token(form='주', tag='VX', start=23, len=1), 
  Token(form='어', tag='EF', start=23, len=1)
 ])]
```
</details>
<hr>
<details>
<summary><code>glue(text_chunks, return_space_insertions=False)</code></summary>
여러 텍스트 조각을 하나로 합치되, 문맥을 고려해 적절한 공백을 사이에 삽입합니다.
이 기능은 OCR로 생성되거나 PDF 등에서 복사하여 강제 개행이 포함된 텍스트를 이어 붙이는데에 용이합니다.

* `text_chunks`: 합칠 텍스트 조각들의 목록입니다.
* `return_space_insertions`: True인 경우, 각 조각별 공백 삽입 유무를 `List[bool]`로 반환합니다.

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
</details>
<hr>

<details>
<summary><code>space(text, reset_whitespace=False)</code></summary>
입력 텍스트에서 띄어쓰기를 교정하여 반환합니다.

* `text`: 분석할 문자열입니다.  이 인자를 단일 str로 줄 경우, 싱글스레드에서 처리하며 str의 Iterable로 줄 경우, 멀티스레드로 분배하여 처리합니다.
* `reset_whitespace` True인 경우 이미 띄어쓰기된 부분을 붙이는 교정도 적극적으로 수행합니다. 기본값은 False로, 이 경우에는 붙어 있는 단어를 띄어쓰는 교정 위주로 수행합니다.

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
</details>
<hr>

<details>
<summary><code>join(morphs, lm_search=True)</code></summary>
형태소들을 결합하여 문장으로 복원합니다. 조사나 어미는 앞 형태소에 맞춰 적절한 형태로 변경됩니다.

* `morphs`: 결합할 형태소의 목록입니다.  각 형태소는 `Kiwi.tokenizer`에서 얻어진 `Token` 타입이거나,  (형태, 품사)로 구성된 `tuple` 타입이어야 합니다.
* `lm_search`: 둘 이상의 형태로 복원 가능한 모호한 형태소가 있는 경우, 이 값이 True면 언어 모델 탐색을 통해 최적의 형태소를 선택합니다. False일 경우 탐색을 실시하지 않지만 더 빠른 속도로 복원이 가능합니다.


이 메소드는 형태소를 결합할 때 `space`에서 사용하는 것과 유사한 규칙을 사용하여 공백을 적절히 삽입합니다.
형태소 그 자체에는 공백 관련 정보가 포함되지 않으므로
특정 텍스트를 `tokenize`로 분석 후 다시 `join`으로 결합하여도 원본 텍스트가 그대로 복원되지는 않습니다.


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

# 0.15.2버전부터는 Tuple의 세번째 요소로 띄어쓰기 유무를 지정할 수 있습니다. 
# True일 경우 강제로 띄어쓰기, False일 경우 강제로 붙여쓰기를 수행합니다.
>> kiwi.join([('길', 'NNG'), ('을', 'JKO', True), ('묻', 'VV'), ('어요', 'EF')])
'길 을 물어요'
>> kiwi.join([('길', 'NNG'), ('을', 'JKO'), ('묻', 'VV', False), ('어요', 'EF')])
'길을물어요'

# 과거형 선어말어미를 제거하는 예시
>> remove_past = lambda s: kiwi.join(t for t in kiwi.tokenize(s) if t.tagged_form != '었/EP')
>> remove_past('먹었다')
'먹다'
>> remove_past('먼 길을 걸었다')
'먼 길을 걷다'
>> remove_past('전화를 걸었다.')
'전화를 걸다.'
```
</details>
<hr>
<details>
<summary><code>template(format_str, cache=True)</code></summary>
형태소들을 결합하여 문장으로 복원합니다. 조사나 어미는 앞 형태소에 맞춰 적절한 형태로 변경됩니다.

* `format_str`: 템플릿 문자열입니다. Python의 `str.format`(https://docs.python.org/ko/3/library/string.html#formatstrings )과 동일한 문법을 사용합니다.
* `cache`: 템플릿의 캐시 여부입니다.

이 메소드는 다음과 같이 `Kiwi.join`의 형태소 결합 기능을 더욱 간편하게 사용할 수 있게 도와줍니다.

```python
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
</details>
<hr>

## 품사 태그

세종 품사 태그를 기초로 하되, 일부 품사 태그를 추가/수정하여 사용하고 있습니다.

<table class='sp'>
<tr><th>대분류</th><th>태그</th><th>설명</th></tr>
<tr><th rowspan='5'>체언(N)</th><td>NNG</td><td>일반 명사</td></tr>
<tr><td>NNP</td><td>고유 명사</td></tr>
<tr><td>NNB</td><td>의존 명사</td></tr>
<tr><td>NR</td><td>수사</td></tr>
<tr><td>NP</td><td>대명사</td></tr>
<tr><th rowspan='5'>용언(V)</th><td>VV</td><td>동사</td></tr>
<tr><td>VA</td><td>형용사</td></tr>
<tr><td>VX</td><td>보조 용언</td></tr>
<tr><td>VCP</td><td>긍정 지시사(이다)</td></tr>
<tr><td>VCN</td><td>부정 지시사(아니다)</td></tr>
<tr><th rowspan='1'>관형사</th><td>MM</td><td>관형사</td></tr>
<tr><th rowspan='2'>부사(MA)</th><td>MAG</td><td>일반 부사</td></tr>
<tr><td>MAJ</td><td>접속 부사</td></tr>
<tr><th rowspan='1'>감탄사</th><td>IC</td><td>감탄사</td></tr>
<tr><th rowspan='9'>조사(J)</th><td>JKS</td><td>주격 조사</td></tr>
<tr><td>JKC</td><td>보격 조사</td></tr>
<tr><td>JKG</td><td>관형격 조사</td></tr>
<tr><td>JKO</td><td>목적격 조사</td></tr>
<tr><td>JKB</td><td>부사격 조사</td></tr>
<tr><td>JKV</td><td>호격 조사</td></tr>
<tr><td>JKQ</td><td>인용격 조사</td></tr>
<tr><td>JX</td><td>보조사</td></tr>
<tr><td>JC</td><td>접속 조사</td></tr>
<tr><th rowspan='5'>어미(E)</th><td>EP</td><td>선어말 어미</td></tr>
<tr><td>EF</td><td>종결 어미</td></tr>
<tr><td>EC</td><td>연결 어미</td></tr>
<tr><td>ETN</td><td>명사형 전성 어미</td></tr>
<tr><td>ETM</td><td>관형형 전성 어미</td></tr>
<tr><th rowspan='1'>접두사</th><td>XPN</td><td>체언 접두사</td></tr>
<tr><th rowspan='4'>접미사(XS)</th><td>XSN</td><td>명사 파생 접미사</td></tr>
<tr><td>XSV</td><td>동사 파생 접미사</td></tr>
<tr><td>XSA</td><td>형용사 파생 접미사</td></tr>
<tr><td>XSM</td><td>부사 파생 접미사<sup>*</sup></td></tr>
<tr><th rowspan='1'>어근</th><td>XR</td><td>어근</td></tr>
<tr><th rowspan='12'>부호, 외국어, 특수문자(S)</th><td>SF</td><td>종결 부호(. ! ?)</td></tr>
<tr><td>SP</td><td>구분 부호(, / : ;)</td></tr>
<tr><td>SS</td><td>인용 부호 및 괄호(' " ( ) [ ] < > { } ― ‘ ’ “ ” ≪ ≫ 등)</td></tr>
<tr><td>SSO</td><td>SS 중 여는 부호<sup>*</sup></td></tr>
<tr><td>SSC</td><td>SS 중 닫는 부호<sup>*</sup></td></tr>
<tr><td>SE</td><td>줄임표(…)</td></tr>
<tr><td>SO</td><td>붙임표(- ~)</td></tr>
<tr><td>SW</td><td>기타 특수 문자</td></tr>
<tr><td>SL</td><td>알파벳(A-Z a-z)</td></tr>
<tr><td>SH</td><td>한자</td></tr>
<tr><td>SN</td><td>숫자(0-9)</td></tr>
<tr><td>SB</td><td>순서 있는 글머리(가. 나. 1. 2. 가) 나) 등)<sup>*</sup></td></tr>
<tr><th rowspan='1'>분석 불능</th><td>UN</td><td>분석 불능<sup>*</sup></td></tr>
<tr><th rowspan='6'>웹(W)</th><td>W_URL</td><td>URL 주소<sup>*</sup></td></tr>
<tr><td>W_EMAIL</td><td>이메일 주소<sup>*</sup></td></tr>
<tr><td>W_HASHTAG</td><td>해시태그(#abcd)<sup>*</sup></td></tr>
<tr><td>W_MENTION</td><td>멘션(@abcd)<sup>*</sup></td></tr>
<tr><td>W_SERIAL</td><td>일련번호(전화번호, 통장번호, IP주소 등)<sup>*</sup></td></tr>
<tr><td>W_EMOJI</td><td>이모지<sup>*</sup></td></tr>
<tr><th rowspan='2'>기타</th><td>Z_CODA</td><td>덧붙은 받침<sup>*</sup></td></tr>
<tr><td>USER0~4</td><td>사용자 정의 태그<sup>*</sup></td></tr>
</table>

<sup>*</sup> 세종 품사 태그와 다른 독자적인 태그입니다.

0.12.0 버전부터 `VV`, `VA`, `VX`, `XSA` 태그에 불규칙 활용여부를 명시하는 접미사 `-R`와 `-I`이 덧붙을 수 있습니다.
`-R`은 규칙 활용,`-I`은 불규칙 활용을 나타냅니다.

## 문장 분리 기능
0.10.3 버전부터 문장 분리 기능을 실험적으로 지원합니다. 0.11.0 버전부터는 정확도가 제법 향상되었습니다. 문장 분리 기능의 성능에 대해서는 [이 페이지](benchmark/sentence_split)를 참조해주세요. 

## 모호성 해소 성능
한 단어가 여러 가지로 형태소 분석이 가능하여 맥락을 보는 게 필수적인 상황에서 Kiwi가 높은 정확도를 보이는 것이 확인되었습니다. 
모호성 해소 성능에 대해서는 [이 페이지](benchmark/disambiguate)를 참조해주세요. 

## 인용하기
인용 방법에 대해서는 [Kiwi#인용하기](https://github.com/bab2min/Kiwi#%EC%9D%B8%EC%9A%A9%ED%95%98%EA%B8%B0)를 참조해주세요.
