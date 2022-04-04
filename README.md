# Python3용 Kiwi

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
```

## 시작하기

kiwipiepy 패키지 설치가 성공적으로 완료되었다면, 다음과 같이 패키지를 import후 Kiwi 객체를 생성했을때 오류가 발생하지 않습니다.
```python
from kiwipiepy import Kiwi, Match
kiwi = Kiwi()
```
Kiwi 생성자는 다음과 같습니다.
```python
Kiwi(num_workers=0, model_path=None, load_default_dict=True, integrate_allomorph=True)
```
* `num_workers`:  2 이상이면 단어 추출 및 형태소 분석에 멀티 코어를 활용하여 조금 더 빠른 속도로 분석을 진행할 수 있습니다. <br>
1인 경우 단일 코어만 활용합니다. num_workers가 0이면 현재 환경에서 사용가능한 모든 코어를 활용합니다. <br>
생략 시 기본값은 0입니다.
* `model_path`: 형태소 분석 모델이 있는 경로를 지정합니다. 생략시 `kiwipiepy_model` 패키지로부터 모델 경로를 불러옵니다.
* `load_default_dict`: 추가 사전을 로드합니다. 추가 사전은 위키백과의 표제어 타이틀로 구성되어 있습니다. 이 경우 로딩 및 분석 시간이 약간 증가하지만 다양한 고유명사를 좀 더 잘 잡아낼 수 있습니다. 분석 결과에 원치 않는 고유명사가 잡히는 것을 방지하려면 이를 False로 설정하십시오.
* `integrate_allomorph`: 어미 중, '아/어', '았/었'과 같이 동일하지만 음운 환경에 따라 형태가 달라지는 이형태들을 자동으로 통합합니다.

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

* `tag` 추가할 형태소들의 품사
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

실제 예시에 대해서는 [Kiwi에 내장된 기본 사전 파일](https://raw.githubusercontent.com/bab2min/Kiwi/main/ModelGenerator/default.dict)을 참조해주세요.
</details>
<hr>

### 분석

kiwi을 생성하고, 사용자 사전에 단어를 추가하는 작업이 완료되었으면 다음 메소드를 사용하여 형태소 분석을 수행할 수 있습니다.

```python
Kiwi.tokenize(text, match_option, normalize_coda)
Kiwi.analyze(text, top_n, match_option, normalize_coda)
``` 

<details>
<summary><code>tokenize(text, match_option=Match.ALL, normalize_coda=False)</code></summary>
입력된 `text`를 형태소 분석하여 그 결과를 간단하게 반환합니다. 분석결과는 다음과 같이 `Token`의 리스트 형태로 반환됩니다.

```python
>> kiwi.tokenize('테스트입니다.')
[Token(form='테스트', tag='NNG', start=0, len=3), Token(form='이', tag='VCP', start=3, len=1), Token(form='ᆸ니다', tag='EF', start=4, len=2)]
```

`normalize_coda`는 ㅋㅋㅋ,ㅎㅎㅎ와 같은 초성체가 뒤따라와서 받침으로 들어갔을때 분석에 실패하는 문제를 해결해줍니다.
```python
>> kiwi.tokenizer("안 먹었엌ㅋㅋ", normalize_coda=False)
[Token(form='안', tag='NNP', start=0, len=1), 
 Token(form='먹었엌', tag='NNP', start=2, len=3), 
 Token(form='ㅋㅋ', tag='SW', start=5, len=2)]
>> kiwi.tokenizer("안 먹었엌ㅋㅋ", normalize_coda=True)
[Token(form='안', tag='MAG', start=0, len=1), 
 Token(form='먹', tag='VV', start=2, len=1), 
 Token(form='었', tag='EP', start=3, len=1), 
 Token(form='어', tag='EF', start=4, len=1), 
 Token(form='ㅋㅋㅋ', tag='SW', start=5, len=2)]
```
</details>
<hr>

<details>
<summary><code>analyze(text, top_n=1, match_option=Match.ALL, normalize_coda=False)</code></summary>
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

## 품사 태그

세종 품사 태그를 기초로 하되, 일부 품사 태그를 추가/수정하여 사용하고 있습니다.

<table>
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
<tr><th rowspan='3'>접미사(XS)</th><td>XSN</td><td>명사 파생 접미사</td></tr>
<tr><td>XSV</td><td>동사 파생 접미사</td></tr>
<tr><td>XSA</td><td>형용사 파생 접미사</td></tr>
<tr><th rowspan='1'>어근</th><td>XR</td><td>어근</td></tr>
<tr><th rowspan='9'>부호, 외국어, 특수문자(S)</th><td>SF</td><td>종결 부호(. ! ?)</td></tr>
<tr><td>SP</td><td>구분 부호(, / : ;)</td></tr>
<tr><td>SS</td><td>인용 부호 및 괄호(' " ( ) [ ] < > { } ― ‘ ’ “ ” ≪ ≫ 등)</td></tr>
<tr><td>SE</td><td>줄임표(…)</td></tr>
<tr><td>SO</td><td>붙임표(- ~)</td></tr>
<tr><td>SW</td><td>기타 특수 문자</td></tr>
<tr><td>SL</td><td>알파벳(A-Z a-z)</td></tr>
<tr><td>SH</td><td>한자</td></tr>
<tr><td>SN</td><td>숫자(0-9)</td></tr>
<tr><th rowspan='1'>분석 불능</th><td>UN</td><td>분석 불능<sup>*</sup></td></tr>
<tr><th rowspan='4'>웹(W)</th><td>W_URL</td><td>URL 주소<sup>*</sup></td></tr>
<tr><td>W_EMAIL</td><td>이메일 주소<sup>*</sup></td></tr>
<tr><td>W_HASHTAG</td><td>해시태그(#abcd)<sup>*</sup></td></tr>
<tr><td>W_MENTION</td><td>멘션(@abcd)<sup>*</sup></td></tr>
</table>

<sup>*</sup> 세종 품사 태그와 다른 독자적인 태그입니다.

## 문장 분리 기능
0.10.3 버전부터 문장 분리 기능을 실험적으로 지원합니다. 0.11.0 버전부터는 정확도가 제법 향상되었습니다. 문장 분리 기능의 성능에 대해서는 [이 페이지](benchmark/sentence_split)를 참조해주세요. 

## 모호성 해소 성능
여러 가지로 형태소 분석이 가능하여 맥락을 보는 게 필수적인 상황에서 Kiwi가 높은 정확도를 보이는 것이 확인되었습니다. 
모호성 해소 성능에 대해서는 [이 페이지](benchmark/disambiguate)를 참조해주세요. 