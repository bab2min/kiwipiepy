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

단, 현재 kiwipiepy 패키지는 Vista 버전 이상의 Windows OS 및 Linux, macOS 10.12 이상을 지원합니다.

0.6.2 이하의 Windows 버전까지는 Python 3.6만 지원하는 문제가 있었으나, 
0.6.3 버전부터는 Python 3.6, 3.7을 바이너리 wheel 형태로 지원하므로 pip로 쉽게 설치하여 사용하실 수 있습니다.
기타 버전의 Python에 대해서도 wheel를 추후 지원할 예정입니다.

0.6.5 버전부터 macOS 10.12 이상에서 clang을 이용한 컴파일 설치가 가능합니다. gcc 혹은 xcode가 설치된 시스템에서 pip를 이용해 Kiwi를 설치할 수 있습니다.
0.9.0 버전부터 Python 3.9를 지원합니다.

## 테스트해보기

Kiwi 0.6.3 버전부터는 설치 후 바로 테스트할 수 있도록 대화형 인터페이스를 지원합니다. pip를 통해 설치가 완료된 후 다음과 같이 실행하여 형태소 분석기를 시험해볼 수 있습니다.

    python -m kiwipiepy

또는

    python3 -m kiwipiepy
	
대화형 인터페이스가 시작되면, 원하는 문장을 입력해 바로 형태소 분석결과를 확인할 수 있습니다.

    >> 안녕?
	[('안녕', 'IC', 0, 2), ('?', 'SF', 2, 3)]

인터페이스를 종료하려면 Ctrl + C 를 누르십시오.

Kiwi에서 사용하는 품사 태그는 세종 말뭉치의 품사 태그를 기초로 하고 일부 태그들을 개량하여 사용하고 있습니다. 자세한 태그 체계에 대해서는 [여기](https://github.com/bab2min/kiwipiepy#%ED%92%88%EC%82%AC-%ED%83%9C%EA%B7%B8)를 참조하십시오.
	
## 시작하기

kiwipiepy 패키지 설치가 성공적으로 완료되었다면, 다음과 같이 패키지를 import후 Kiwi 객체를 생성했을때 오류가 발생하지 않습니다.
```python
from kiwipiepy import Kiwi, Option
kiwi = Kiwi()
```
Kiwi 생성자는 다음과 같습니다.
```python
Kiwi(num_workers=0, model_path='./', options=Option.LOAD_DEFAULT_DICTIONARY | Option.INTEGRATE_ALLOMORPH)
```
num_workers가 2 이상이면 단어 추출 및 형태소 분석에 멀티 코어를 활용하여 조금 더 빠른 속도로 분석을 진행할 수 있습니다. 
num_workers가 1인 경우 단일 코어만 활용합니다. num_workers가 0이면 현재 환경에서 사용가능한 모든 코어를 활용합니다. 
생략 시 기본값은 0입니다.

model_path는 형태소 분석 모델이 있는 경로를 지정합니다.

options은 Kiwi 실행시에 설정한 다양한 옵션들을 세팅하는 비트 마스크 값입니다. 사용 가능한 값은 다음과 같습니다.

* `Option.LOAD_DEFAULT_DICTIONARY` : 추가 사전을 로드합니다. 추가 사전은 위키백과의 표제어 타이틀로 구성되어 있습니다. 이 경우 로딩 및 분석 시간이 약간 증가하지만 다양한 고유명사를 좀 더 잘 잡아낼 수 있습니다.
* `Option.INTEGRATE_ALLOMORPH` : 어미 중, '아/어', '았/었'과 같이 동일하지만 음운 환경에 따라 형태가 달라지는 이형태들을 자동으로 통합합니다.

kiwi 객체는 크게 다음 세 종류의 작업을 수행할 수 있습니다.
* 코퍼스로부터 미등록 단어 추출
* 사용자 사전 추가
* 형태소 분석

## 코퍼스로부터 미등록 단어 추출

Kiwi 0.5부터 새로 추가된 기능입니다. 자주 등장하는 문자열의 패턴을 파악하여 단어로 추정되는 문자열을 추출해줍니다. 
이 기능의 기초적인 아이디어는 https://github.com/lovit/soynlp 의 Word Extraction 기법을 바탕으로 하고 있으며, 
이에 문자열 기반의 명사 확률을 조합하여 명사일 것으로 예측되는 단어만 추출합니다.

Kiwi가 제공하는 미등록 단어 추출 관련 메소드는 다음 세 가지입니다.
```python
kiwi.extract_words(reader, min_cnt, max_word_len, min_score)
kiwi.extract_filter_words(reader, min_cnt, max_word_len, min_score, pos_score)
kiwi.extract_add_words(reader, min_cnt, max_word_len, min_score, pos_score)
```
**`extract_words(reader, min_cnt=10, max_word_len=10, min_score=0.25)`**

reader가 읽어들인 텍스트로부터 단어 후보를 추출합니다. 
reader는 다음과 같은 호출가능한(Callable) 객체여야합니다.
min_cnt는 추출할 단어가 입력 텍스트 내에서 최소 몇 번 이상 등장하는 지를 결정합니다. 입력 텍스트가 클 수록 그 값을 높여주시는게 좋습니다.
max_word_len는 추출할 단어의 최대 길이입니다. 이 값을 너무 크게 설정할 경우 단어를 스캔하는 시간이 길어지므로 적절하게 조절해주시는 게 좋습니다.
min_score는 추출할 단어의 최소 단어 점수입니다. 이 값을 낮출수록 단어가 아닌 형태가 추출될 가능성이 높아지고, 
반대로 이 값을 높일 수록 추출되는 단어의 개수가 줄어들므로 적절한 수치로 설정하실 필요가 있습니다. 기본값은 0.25입니다.
```python
class ReaderExam:
  def __init__(self, filePath):
    self.file = open(filePath)

  def read(self, id):
    if id == 0: self.file.seek(0)
    return self.file.readline()

reader = ReaderExam('test.txt')
kiwi.extract_words(reader.read, 10, 10, 0.25)
```
reader는 첫번째 인자로 id를 받습니다. id는 현재 읽어야할 행의 번호를 알려주며, id == 0인 경우 파일을 처음부터 다시 읽어야합니다. 
extract계열의 함수는 단어 후보를 추출하는 과정에서 입력 텍스트 파일을 여러 번 다시 읽으므로, 
id == 0인 경우를 적절하게 처리해주어야 올바른 결과를 얻으실 수 있습니다.

**`extract_filter_words(reader, min_cnt=10, max_word_len=10, min_score=0.25, pos_score=-3)`**

extractWords와 동일하게 단어 후보를 추출하고, 그 중에서 이미 형태소 분석 사전에 등록된 경우 및 명사가 아닐 것으로 예측되는 단어 후보를 제거하여
실제 명사로 예측되는 단어 목록만 추출해줍니다. 
pos_score는 추출할 단어의 최소 명사 점수입니다. 이 값을 낮출수록 명사가 아닌 단어들이 추출될 가능성이 높으며, 
반대로 높일수록 추출되는 명사의 개수가 줄어듭니다. 기본값은 -3입니다.
나머지 인자는 extractWords와 동일합니다. 

**`extract_add_words(reader, min_cnt=10, max_word_len=10, min_score=0.25, pos_score=-3)`**

extractFilterWords와 동일하게 명사인 단어만 추출해줍니다. 
다만 이 메소드는 추출된 명사 후보를 자동으로 사용자 사전에 등록하여 형태소 분석에 사용할 수 있게 해줍니다. 
만약 이 메소드를 사용하지 않으신다면 add_user_word 메소드를 사용하여 추출된 미등록 단어를 직접 사용자 사전에 등록하셔야 합니다.

## 사용자 사전 추가

기존의 사전에 등록되지 않은 단어를 제대로 분석하기 위해서는 사용자 사전에 해당 단어를 등록해주어야 합니다. 
이는 extract_add_words를 통해서 자동으로 이뤄질 수도 있고, 수작업으로 직접 추가될 수도 있습니다. 
다음 메소드들은 사용자 사전을 관리하는데 사용되는 메소드들입니다.
```python
kiwi.add_user_word(word, pos, score)
kiwi.load_user_dictionary(userDictPath)
```
**`add_user_word(word, pos='NNP', score=0.0)`**

사용자 사전에 word를 등록합니다. 현재는 띄어쓰기(공백문자)가 포함되지 않는 문자열만 단어로 등록할 수 있습니다.
pos는 등록할 단어의 품사입니다. 세종 품사태그를 따르며, 기본값은 NNP(고유명사)입니다.
score는 등록할 단어의 점수입니다. 
동일한 형태라도 여러 경우로 분석될 가능성이 있는 경우에, 이 값이 클수록 해당 단어가 더 우선권을 가지게 됩니다.


**`load_user_dictionary(userDictPath)`**

파일로부터 사용자 사전을 읽어들입니다. 사용자 사전 파일은 UTF-8로 인코딩되어 있어야하며, 다음과 같은 형태로 구성되어야 합니다.
탭 문자(\t)로 각각의 필드는 분리되어야 하며, 단어 점수는 생략 가능합니다. 

    #주석은 #으로 시작합니다.
    단어1 [탭문자] 품사태그 [탭문자] 단어점수
    단어2 [탭문자] 품사태그 [탭문자] 단어점수
    단어3 [탭문자] 품사태그 [탭문자] 단어점수

실제 예시

    # 스타크래프트 관련 명사 목록
    스타크래프트  NNP 3.0
    저글링 NNP
    울트라리스크 NNP  3.0
    
# 형태소 분석

kiwi을 생성하고, 사용자 사전에 단어를 추가하는 작업이 완료되었으면 prepare를 호출하여 분석 모델을 준비해야합니다.

**`kiwi.prepare()`**

이 메소드는 별다른 파라미터를 필요로하지 않으며, 성공하였을 경우 0, 실패하였을 경우 0이 아닌 값을 돌려줍니다.

실제 형태소를 분석하는 메소드에는 다음이 있습니다.
```python
kiwi.analyze(text, top_n)
kiwi.analyze(reader, receiver, top_n) # 0.10.0에서 제거될 예정
``` 

**`analyze(text, top_n=1)`**

입력된 text를 형태소 분석하여 그 결과를 반환합니다. 총 top_n개의 결과를 출력합니다. 반환값은 다음과 같이 구성됩니다.
```
[(분석결과1, 점수), (분석결과2, 점수), ... ]
```
분석결과는 다음과 같이 튜플의 리스트 형태로 반환됩니다.
```
[(단어1, 품사태그, 단어 시작 지점, 단어 길이), (단어2, 품사태그, 단어 시작 지점, 단어 길이), ...]
```
실제 예시는 다음과 같습니다.
```
>> kiwi.analyze('테스트입니다.', 5)
[([('테스트', 'NNG', 0, 3), ('이', 'VCP', 3, 1), ('ᆸ니다', 'EF', 4, 2), ('.', 'SF', 6, 1)], -20.393310546875), 
 ([('테스 트입니', 'NNG', 0, 5), ('다', 'EF', 5, 1), ('.', 'SF', 6, 1)], -29.687143325805664), 
 ([('테스트입니', 'NNG', 0, 5), ('이', 'VCP', 5, 1), ('다', 'EF', 5, 1), ('.', 'SF', 6, 1)], -31.221078872680664), 
 ([('테스트입니', 'NNP', 0, 5), ('이', 'VCP', 5, 1), ('다', 'EF', 5, 1), ('.', 'SF', 6, 1)], -32.20524978637695), 
 ([('테스트', 'NNG', 0, 3), ('이', 'MM', 3, 1), ('ᆸ니다', 'EF', 4, 2), ('.', 'SF', 6, 1)], -32.859375)]
```
만약 text가 str의 iterable인 경우 여러 개의 입력을 병렬로 처리합니다. 이때의 반환값은 단일 text를 입력한 경우의 반환값의 iterable입니다.
Kiwi() 생성시 인자로 준 num_workers에 따라 여러 개의 스레드에서 작업이 동시에 처리됩니다. 반환되는 값은 입력되는 값의 순서와 동일합니다.
```
>> result_iter = kiwi.analyze(['테스트입니다.', '테스트가 아닙니다.', '사실 맞습니다.'])
>> next(result_iter)
[([('테스트', 'NNG', 0, 3), ('이', 'VCP', 3, 1), ('ᆸ니다', 'EF', 4, 2), ('.', 'SF', 6, 1)], -20.393310546875)]
>> next(result_iter)
[([('테스트', 'NNG', 0, 3), ('가', 'JKC', 3, 1), ('아니', 'VCN', 5, 2), ('ᆸ니다', 'EF', 7, 2), ('.', 'SF', 9, 1)], -30.220947265625)]
>> next(result_iter)
[([('사실', 'MAG', 0, 2), ('맞', 'VV', 3, 1), ('습니다', 'EF', 4, 3), ('.', 'SF', 7, 1)], -22.192138671875)]
>> next(result_iter)
Traceback (most recent call last):
    File "<stdin>", line 1, in <module>
StopIteration
```
for 반복문을 사용하면 좀더 간단하고 편리하게 병렬 처리를 수행할 수 있습니다. 이는 대량의 텍스트 데이터를 분석할 때 유용합니다.
```
>> for result in kiwi.analyze(long_list_of_text):
      tokens, score = result[0]
      print(tokens)
```
text를 str의 iterable로 준 경우 이 iterable을 읽어들이는 시점은 analyze 호출 이후일 수도 있습니다. 
따라서 이 인자가 다른 IO 자원(파일 입출력 등)과 연동되어 있다면 모든 분석이 끝나기 전까지 해당 자원을 종료하면 안됩니다.
```
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
**`analyze(reader, receiver, top_n = 1)`**
 
analyze 메소드는 또 다른 형태로도 호출할 수 있습니다. reader와 receiver를 사용해 호출하는 경우,
문자열 읽기/쓰기 부분과 분석 부분이 별도의 스레드에서 동작하며, 여분의 스레드가 있을 경우 분석 역시 멀티 코어를 활용하여 성능 향상을 꽤할 수 있습니다.
이 방법은 Pythonic하지 않기에 더 이상 권장되지 않습니다. 0.10.0 버전에서 이 기능은 제거될 예정입니다. 
대신 위에서 소개한 analyze()를 사용하기를 권장합니다.
 
reader와 receiver를 사용한 예시는 다음과 같습니다.
 ```python
from kiwipiepy import Kiwi

class IOHandler:
  def __init__(self, input, output):
    self.input = open(input, encoding='utf-8')
    self.output = open(output, 'w', encoding='utf-8')

  def read(self, sent_id):
    if sent_id == 0:
      self.input.seek(0)
      self.iter = iter(self.input)
    try:
      return next(self.iter)
    except StopIteration:
      return None

  def write(self, sent_id, res):
    print('Analyzed %dth row' % sent_id)
    self.output.write(' '.join(map(lambda x:x[0]+'/'+x[1], res[0][0])) + '\n')

  def __del__(self):
    self.input.close()
    self.output.close()

kiwi = Kiwi()
kiwi.load_user_dictionary('userDict.txt')
kiwi.prepare()
handle = IOHandler('test.txt', 'result.txt')
kiwi.analyze(handle.read, handle.write)
```
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

# 외부 라이브러리 라이센스

* [mimalloc](https://github.com/microsoft/mimalloc) : [MIT License](https://github.com/microsoft/mimalloc/blob/master/LICENSE)
