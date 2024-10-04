Kiwipiepy란?
============
Kiwipiepy는 한국어 형태소 분석기인 Kiwi(Korean Intelligent Word Identifier)의 Python 모듈입니다. 
C++로 작성되었고 다른 패키지에 의존성이 없으므로 C++ 컴파일이 가능한 환경이라면 어디에서나 Kiwipiepy를 사용 가능합니다.

![kiwipiepy](https://badge.fury.io/py/kiwipiepy.svg)

시작하기
--------
pip를 이용해 쉽게 설치할 수 있습니다. (https://pypi.org/project/kiwipiepy/)

```bash
$ pip install kiwipiepy
```
지원하는 OS와 Python 버전은 다음과 같습니다:

* Python 3.7 이상이 설치된 Linux (x86-64) 
* Python 3.7 이상이 설치된 macOS 10.13이나 그 이후 버전
* Python 3.7 이상이 설치된 Windows 7 이나 그 이후 버전 (x86, x86-64)
* Python 3.7 이상이 설치된 다른 OS: 이 경우 소스 코드 컴파일을 위해 C++11이 지원되는 컴파일러가 필요합니다.

Kiwipiepy가 제대로 설치되었는지 확인하기 위해서는 다음 명령어를 실행해보십시오.

```bash
$ python -m kiwipiepy
```
위 명령어는 대화형 인터페이스를 시작합니다. 인터페이스에 원하는 문장을 입력하면 형태소 분석 결과를 확인할 수 있습니다.

```text
>> 안녕?
[Token(form='안녕', tag='IC', start=0, len=2), Token(form='?', tag='SF', start=2, len=3)]
```
인터페이스를 종료하려면 Ctrl + C 를 누르십시오.

예제
----
**간단한 분석**

다음 예제 코드는 kiwipiepy 인스턴스를 생성해 형태소 분석을 수행하는 간단한 예제 코드입니다.

```python
from kiwipiepy import Kiwi
kiwi = Kiwi()
for result, score in kiwi.analyze("형태소 분석 결과입니다", top_n=5):
    print(score, result, sep='\t')

# 위 코드를 실행하면 다음과 같은 결과가 나옵니다.
# -34.33329391479492      [Token(form='형태소', tag='NNG', start=0, len=3), Token(form='분석', tag='NNG', start=4, len=2), Token(form='결과', tag='NNG', start=7, len=2), Token(form='이', tag='VCP', start=9, len=1), Token(form='ᆸ니다', tag='EF', start=10, len=2)]
# -38.10548400878906      [Token(form='형태소', tag='NNG', start=0, len=3), Token(form='분석', tag='NNG', start=4, len=2), Token(form='결과', tag='NNG', start=7, len=2), Token(form='이', tag='MM', start=9, len=1), Token(form='ᆸ니다', tag='EC', start=10, len=2)]
# -51.977012634277344     [Token(form='형태소', tag='NNG', start=0, len=3), Token(form='분석', tag='NNG', start=4, len=2), Token(form='결과', tag='NNG', start=7, len=2), Token(form='이', tag='MM', start=9, len=1), Token(form='ᆸ니다', tag='NNP', start=10, len=2)]
# -51.978363037109375     [Token(form='형태소', tag='NNG', start=0, len=3), Token(form='분석', tag='NNG', start=4, len=2), Token(form='결과', tag='NNG', start=7, len=2), Token(form='이', tag='MM', start=9, len=1), Token(form='ᆸ', tag='NNG', start=10, len=0), Token(form='니', tag='EC', start=10, len=1), Token(form='다', tag='EC', start=11, len=1)]
# -52.152374267578125     [Token(form='형태소', tag='NNG', start=0, len=3), Token(form='분석', tag='NNG', start=4, len=2), Token(form='결과', tag='NNG', start=7, len=2), Token(form='이', tag='MM', start=9, len=1), Token(form='ᆸ', tag='NNG', start=10, len=0), Token(form='니다', tag='EF', start=10, len=2)]

# 간단하게 형태소 분석 결과만 얻고 싶다면 `tokenize` 메소드를 사용하면 됩니다.

result = kiwi.tokenize("형태소 분석 결과입니다")
print(result)
# [Token(form='형태소', tag='NNG', start=0, len=3), Token(form='분석', tag='NNG', start=4, len=2), Token(form='결과', tag='NNG', start=7, len=2), Token(form='이', tag='VCP', start=9, len=1), Token(form='ᆸ니다', tag='EF', start=10, len=2)]
```

**사용자 단어 추가**

사용자 정의 단어를 추가하여 형태소 분석을 수행하는 예제입니다. 사용자 정의 단어를 등록하면 이는 Kiwi 분석기의 사전에 포함되어 결과의 후보로 등장할 수 있게 됩니다.

종종 동일한 형태의 단어가 여러 가지로 분석되는 경우가 있습니다. 이 경우 사용자 정의 단어를 우선할지, 분석기가 가지고 있는 형태소 정보를 우선할지 사용자 단어 점수를 조절함으로써 통제 가능합니다.
아래 예제는 '골리'라는 고유 명사 단어가 포함된 문장을 분석하는 경우에 부여하는 단어 점수에 따라 결과가 어떻게 달라지는지를 보여줍니다.

```python
from kiwipiepy import Kiwi

# 사용자 단어 추가 없이 분석해보겠습니다.

kiwi = Kiwi()

print(*kiwi.analyze('사람을 골리다', top_n=5), sep='\n')
# 결과
# ([('사람', 'NNG', 0, 2), ('을', 'JKO', 2, 1), ('골리', 'VV', 4, 2), ('다', 'EC', 6, 1)], -36.505615234375)
# ([('사람', 'NNG', 0, 2), ('을', 'JKO', 2, 1), ('골리', 'VV', 4, 2), ('다', 'MAG', 6, 1)], -40.310791015625)
# ([('사람', 'NNG', 0, 2), ('을', 'JKO', 2, 1), ('골리', 'VV', 4, 2), ('하', 'XSA', 6, 1), ('다', 'EC', 6, 1)], -40.388427734375)
# ([('사람', 'NNG', 0, 2), ('을', 'JKO', 2, 1), ('골리', 'VV', 4, 2), ('하', 'XSV', 6, 1), ('다', 'EC', 6, 1)], -42.22119140625)
# ([('사람', 'NNG', 0, 2), ('을', 'JKO', 2, 1), ('골리', 'VV', 4, 2), ('다', 'EF', 6, 1)], -42.44189453125)

print(*kiwi.analyze('골리는 사람이다', top_n=5), sep='\n')
# 결과
# ([('골리', 'VV', 0, 2), ('는', 'ETM', 2, 1), ('사람', 'NNG', 4, 2), ('이', 'VCP', 6, 1), ('다', 'EC', 7, 1)], -39.06201171875)
# ([('골리', 'VV', 0, 2), ('는', 'ETM', 2, 1), ('사람', 'NNG', 4, 2), ('이', 'VCP', 6, 1), ('다', 'EF', 7, 1)], -41.10693359375)
# ([('골리', 'VV', 0, 2), ('는', 'ETM', 2, 1), ('사람', 'NNG', 4, 2), ('이', 'JKS', 6, 1), ('다', 'MAG', 7, 1)], -41.588623046875)
# ([('골리', 'VV', 0, 2), ('는', 'JX', 2, 1), ('사람', 'NNG', 4, 2), ('이', 'VCP', 6, 1), ('다', 'EC', 7, 1)], -41.6220703125)
# ([('골리', 'VV', 0, 2), ('는', 'JX', 2, 1), ('사람', 'NNG', 4, 2), ('이', 'JKS', 6, 1), ('다', 'MAG', 7, 1)], -43.114990234375)

# 사용자 단어 '골리'를 추가해보도록 하겠습니다.
kiwi = Kiwi()
kiwi.add_user_word('골리', 'NNP', 0)

print(*kiwi.analyze('사람을 골리다', top_n=5), sep='\n')
# 결과
# ([('사람', 'NNG', 0, 2), ('을', 'JKO', 2, 1), ('골리', 'NNP', 4, 2), ('다', 'EC', 6, 1)], -31.064453125)
# ([('사람', 'NNG', 0, 2), ('을', 'JKO', 2, 1), ('골리', 'NNP', 4, 2), ('다', 'MAG', 6, 1)], -34.109619140625)
# ([('사람', 'NNG', 0, 2), ('을', 'JKO', 2, 1), ('골리', 'NNP', 4, 2), ('다', 'EF', 6, 1)], -37.097900390625)
# ([('사람', 'NNG', 0, 2), ('을', 'JKO', 2, 1), ('골', 'NNG', 4, 1), ('리다', 'EF', 5, 2)], -45.919189453125)
# ([('사람', 'NNG', 0, 2), ('을', 'JKO', 2, 1), ('골', 'VV', 4, 1), ('리다', 'EF', 5, 2)], -49.18359375)

print(*kiwi.analyze('골리는 사람이다', top_n=5), sep='\n')
# 결과
# ([('골리', 'NNP', 0, 2), ('는', 'JX', 2, 1), ('사람', 'NNG', 4, 2), ('이', 'VCP', 6, 1), ('다', 'EC', 7, 1)], -25.12841796875)
# ([('골리', 'NNP', 0, 2), ('는', 'JX', 2, 1), ('사람', 'NNG', 4, 2), ('이', 'JKS', 6, 1), ('다', 'MAG', 7, 1)], -26.621337890625)
# ([('골리', 'NNP', 0, 2), ('는', 'JX', 2, 1), ('사람', 'NNG', 4, 2), ('이', 'VCP', 6, 1), ('다', 'EF', 7, 1)], -27.17333984375)
# ([('골리', 'NNP', 0, 2), ('는', 'ETM', 2, 1), ('사람', 'NNG', 4, 2), ('이', 'VCP', 6, 1), ('다', 'EC', 7, 1)], -29.90185546875)
# ([('골리', 'NNP', 0, 2), ('는', 'ETM', 2, 1), ('사람', 'NNG', 4, 2), ('이', 'VCP', 6, 1), ('다', 'EF', 7, 1)], -31.94677734375)

# 사용자 단어 '골리'의 점수를 낮춰서 추가해보도록 하겠습니다.
kiwi = Kiwi()
kiwi.add_user_word('골리', 'NNP', -6)

print(*kiwi.analyze('사람을 골리다', top_n=5), sep='\n')
# 결과
# ([('사람', 'NNG', 0, 2), ('을', 'JKO', 2, 1), ('골리', 'VV', 4, 2), ('다', 'EC', 6, 1)], -36.505615234375)
# ([('사람', 'NNG', 0, 2), ('을', 'JKO', 2, 1), ('골리', 'NNP', 4, 2), ('다', 'EC', 6, 1)], -37.064453125)
# ([('사람', 'NNG', 0, 2), ('을', 'JKO', 2, 1), ('골리', 'NNP', 4, 2), ('다', 'MAG', 6, 1)], -40.109619140625)
# ([('사람', 'NNG', 0, 2), ('을', 'JKO', 2, 1), ('골리', 'VV', 4, 2), ('다', 'MAG', 6, 1)], -40.310791015625)
# ([('사람', 'NNG', 0, 2), ('을', 'JKO', 2, 1), ('골리', 'VV', 4, 2), ('다', 'EF', 6, 1)], -42.44189453125)

print(*kiwi.analyze('골리는 사람이다', top_n=5), sep='\n')    
# 결과
# ([('골리', 'NNP', 0, 2), ('는', 'JX', 2, 1), ('사람', 'NNG', 4, 2), ('이', 'VCP', 6, 1), ('다', 'EC', 7, 1)], -31.12841796875)
# ([('골리', 'NNP', 0, 2), ('는', 'JX', 2, 1), ('사람', 'NNG', 4, 2), ('이', 'JKS', 6, 1), ('다', 'MAG', 7, 1)], -32.621337890625)
# ([('골리', 'NNP', 0, 2), ('는', 'JX', 2, 1), ('사람', 'NNG', 4, 2), ('이', 'VCP', 6, 1), ('다', 'EF', 7, 1)], -33.17333984375)
# ([('골리', 'NNP', 0, 2), ('는', 'ETM', 2, 1), ('사람', 'NNG', 4, 2), ('이', 'VCP', 6, 1), ('다', 'EC', 7, 1)], -35.90185546875)
# ([('골리', 'NNP', 0, 2), ('는', 'ETM', 2, 1), ('사람', 'NNG', 4, 2), ('이', 'VCP', 6, 1), ('다', 'EF', 7, 1)], -37.94677734375)
```
**멀티스레딩 analyze**

다음 예제 코드는 멀티스레드를 활용하여 `test.txt` 파일을 줄별로 읽어들여 형태소 분석한 뒤 그 결과를 `result.txt` 에 저장합니다.

```python
from kiwipiepy import Kiwi
# 4개의 스레드에서 동시에 처리합니다.
# num_workers 생략시 현재 환경에서 사용가능한 모든 코어를 다 사용합니다.
kiwi = Kiwi(num_workers=4)
with open('result.txt', 'w', encoding='utf-8') as output:
    for res in kiwi.analyze(open('test.txt', encoding='utf-8')):
        output.write(' '.join(map(lambda x:x[0]+'/'+x[1], res[0][0])) + '\n')
```
`Kiwi()` 생성시 인자로 준 num_workers에 따라 여러 개의 스레드에서 작업이 동시에 처리됩니다. 반환되는 값은 입력되는 값의 순서와 동일합니다.

`analyze` 를 인자를 str의 iterable로 준 경우 이 iterable을 읽어들이는 시점은 analyze 호출 이후일 수도 있습니다. 
따라서 이 인자가 다른 IO 자원(파일 입출력 등)과 연동되어 있다면 모든 분석이 끝나기 전까지 해당 자원을 종료하면 안됩니다.
예를 들어 다음과 같이 open을 통해 생성한 파일 입출력 객체를 미리 종료하는 경우 오류가 발생할 수 있습니다.

```python
from kiwipiepy import Kiwi
kiwi = Kiwi(num_workers=4)
file = open('long_text.txt', encoding='utf-8')
result_iter = kiwi.analyze(file)
file.close() # 파일이 종료됨
next(result_iter) # 종료된 파일에서 분석해야할 다음 텍스트를 읽어들이려고 시도하여 오류 발생

# ValueError: I/O operation on closed file.
# The above exception was the direct cause of the following exception:
# Traceback (most recent call last):
#   File "<stdin>", line 1, in <module>
# SystemError: <built-in function next> returned a result with an error set
```

**normalize_coda**

0.10.2버전부터 `normalize_coda` 기능이 추가되었습니다. 이 기능은 웹이나 채팅 텍스트 데이터에서 자주 쓰이는 
ㅋㅋㅋ, ㅎㅎㅎ와 같은 초성체가 어절 뒤에 붙는 경우 분석에 실패하는 경우를 막아줍니다.

```python
from kiwipiepy import Kiwi
kiwi = Kiwi()
kiwi.tokenize("안 먹었엌ㅋㅋ", normalize_coda=False)
# 출력
 [Token(form='안', tag='NNP', start=0, len=1), 
  Token(form='먹었엌', tag='NNP', start=2, len=3), 
  Token(form='ㅋㅋ', tag='SW', start=5, len=2)]

kiwi.tokenize("안 먹었엌ㅋㅋ", normalize_coda=True)
# 출력
 [Token(form='안', tag='MAG', start=0, len=1), 
  Token(form='먹', tag='VV', start=2, len=1), 
  Token(form='었', tag='EP', start=3, len=1), 
  Token(form='어', tag='EF', start=4, len=1), 
  Token(form='ㅋㅋㅋ', tag='SW', start=5, len=2)]
```

**z_coda**

0.15.0버전부터 `z_coda` 기능이 추가되었습니다. 이 기능은 조사 및 어미에 덧붙은 받침을 분리해줍니다. 
기본적으로 True로 설정되어 있으며, 이 기능을 사용하지 않으려면 인자로 z_coda=False를 주어야 합니다.

```python
from kiwipiepy import Kiwi
kiwi = Kiwi()
kiwi.tokenize('우리집에성 먹었어욥', z_coda=False)
# 출력
 [Token(form='우리', tag='NP', start=0, len=2), 
  Token(form='집', tag='NNG', start=2, len=1), 
  Token(form='에', tag='JKB', start=3, len=1), 
  Token(form='성', tag='NNG', start=4, len=1), 
  Token(form='먹었어욥', tag='NNG', start=6, len=4)]

kiwi.tokenize("우리집에성 먹었어욥", z_coda=True) # 기본값이 True므로 z_coda=True는 생략 가능
# 출력
 [Token(form='우리', tag='NP', start=0, len=2), 
  Token(form='집', tag='NNG', start=2, len=1), 
  Token(form='에서', tag='JKB', start=3, len=2), 
  Token(form='ᆼ', tag='Z_CODA', start=4, len=1), 
  Token(form='먹', tag='VV', start=6, len=1), 
  Token(form='었', tag='EP', start=7, len=1), 
  Token(form='어요', tag='EF', start=8, len=2), 
  Token(form='ᆸ', tag='Z_CODA', start=9, len=1)]
```

**split_complex**

0.15.0버전부터 `split_complex` 기능이 추가되었습니다. 이 기능은 더 잘게 분할 가능한 형태소들을 최대한 분할하도록 합니다.
이런 형태소에는 '고마움(고맙 + 음)'과 같은 파생 명사, '건강히(건강 + 히)'와 같은 파생 부사, '반짝거리다(반짝 + 거리다)', '걸어다니다(걸어 + 다니다)'와 같은 파생 동/형용사 등이 포함됩니다.
```python
from kiwipiepy import Kiwi
kiwi = Kiwi()
kiwi.tokenize('고마움에 건강히 지내시라고 눈을 반짝거리며 인사했다', split_complex=False)
# 출력
 [Token(form='고마움', tag='NNG', start=0, len=3),
  Token(form='에', tag='JKB', start=3, len=1), 
  Token(form='건강히', tag='MAG', start=5, len=3), 
  Token(form='지내', tag='VV', start=9, len=2), 
  Token(form='시', tag='EP', start=11, len=1), 
  Token(form='라고', tag='EC', start=12, len=2), 
  Token(form='눈', tag='NNG', start=15, len=1), 
  Token(form='을', tag='JKO', start=16, len=1), 
  Token(form='반짝거리', tag='VV', start=18, len=4), 
  Token(form='며', tag='EC', start=22, len=1), 
  Token(form='인사', tag='NNG', start=24, len=2), 
  Token(form='하', tag='XSV', start=26, len=1), 
  Token(form='었', tag='EP', start=26, len=1), 
  Token(form='다', tag='EF', start=27, len=1)]

kiwi.tokenize('고마움에 건강히 지내시라고 눈을 반짝거리며 인사했다', split_complex=True)
# 출력
 [Token(form='고맙', tag='VA-I', start=0, len=3), 
  Token(form='음', tag='ETN', start=2, len=1), 
  Token(form='에', tag='JKB', start=3, len=1), 
  Token(form='건강', tag='NNG', start=5, len=2), 
  Token(form='히', tag='XSM', start=7, len=1), 
  Token(form='지내', tag='VV', start=9, len=2), 
  Token(form='시', tag='EP', start=11, len=1), 
  Token(form='라고', tag='EC', start=12, len=2), 
  Token(form='눈', tag='NNG', start=15, len=1), 
  Token(form='을', tag='JKO', start=16, len=1), 
  Token(form='반짝', tag='MAG', start=18, len=2), 
  Token(form='거리', tag='XSV', start=20, len=2), 
  Token(form='며', tag='EC', start=22, len=1), 
  Token(form='인사', tag='NNG', start=24, len=2), 
  Token(form='하', tag='XSV', start=26, len=1), 
  Token(form='었', tag='EP', start=26, len=1), 
  Token(form='다', tag='EF', start=27, len=1)]   

```

**blocklist**

0.15.0부터 `split_complex` 와 더불어 `blocklist` 기능도 추가되었습니다. 이 기능은 `split_complex` 와는 다르게 세부적으로 분석 결과에 등장하면 안되는 형태소 목록을 지정할 수 있습니다.
```python
from kiwipiepy import Kiwi
kiwi = Kiwi()
kiwi.tokenize('고마움에 건강히 지내시라고 눈을 반짝거리며 인사했다')
# 출력
 [Token(form='고마움', tag='NNG', start=0, len=3),
  Token(form='에', tag='JKB', start=3, len=1), 
  Token(form='건강히', tag='MAG', start=5, len=3), 
  Token(form='지내', tag='VV', start=9, len=2), 
  Token(form='시', tag='EP', start=11, len=1), 
  Token(form='라고', tag='EC', start=12, len=2), 
  Token(form='눈', tag='NNG', start=15, len=1), 
  Token(form='을', tag='JKO', start=16, len=1), 
  Token(form='반짝거리', tag='VV', start=18, len=4), 
  Token(form='며', tag='EC', start=22, len=1), 
  Token(form='인사', tag='NNG', start=24, len=2), 
  Token(form='하', tag='XSV', start=26, len=1), 
  Token(form='었', tag='EP', start=26, len=1), 
  Token(form='다', tag='EF', start=27, len=1)]

kiwi.tokenize('고마움에 건강히 지내시라고 눈을 반짝거리며 인사했다', blocklist=['고마움/NNG'])
# 출력
 [Token(form='고맙', tag='VA-I', start=0, len=3), 
  Token(form='음', tag='ETN', start=2, len=1), 
  Token(form='에', tag='JKB', start=3, len=1), 
  Token(form='건강히', tag='MAG', start=5, len=3), 
  Token(form='지내', tag='VV', start=9, len=2), 
  Token(form='시', tag='EP', start=11, len=1), 
  Token(form='라고', tag='EC', start=12, len=2), 
  Token(form='눈', tag='NNG', start=15, len=1), 
  Token(form='을', tag='JKO', start=16, len=1), 
  Token(form='반짝거리', tag='VV', start=18, len=4), 
  Token(form='며', tag='EC', start=22, len=1), 
  Token(form='인사', tag='NNG', start=24, len=2), 
  Token(form='하', tag='XSV', start=26, len=1), 
  Token(form='었', tag='EP', start=26, len=1), 
  Token(form='다', tag='EF', start=27, len=1)]
```

**공백이 포함된 단어의 분석**

0.17.0 버전부터 공백을 포함한 단어를 사전에 추가하는 것이 가능해졌습니다. 사전에 등록된 다어절 단어는 등록한 형태와 동일한 위치에 공백이 있는 경우뿐만 아니라 공백이 없는 경우에도 일치가 됩니다. 그러나 공백이 없는 지점에 공백이 들어간 텍스트에는 일치가 되지 않습니다.
```python
>>> from kiwipiepy import Kiwi
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
```

iterable한 입력을 받는 메소드
--------------------
`analyze`, `tokenize`, `extract_words`, `extract_add_words`는 iterable str을 입력받을 수 있습니다.

**analyze의 사용법**
```python
from kiwipiepy import Kiwi

kiwi = Kiwi()
kiwi.load_user_dictionary('userDict.txt')
with open('result.txt', 'w', encoding='utf-8') as out:
    for res in kiwi.analyze(open('test.txt', encoding='utf-8')):
        score, tokens = res[0] # top-1 결과를 가져옴
        print(*(token.tagged_form for token in tokens), file=out)

# tokenize 메소드를 사용하면 위와 동일한 동작을 수행하되 더 간결한 코드도 가능합니다.
with open('result.txt', 'w', encoding='utf-8') as out:
    for tokens in kiwi.tokenize(open('test.txt', encoding='utf-8')):
        print(*(token.tagged_form for token in tokens), file=out)
```
**extract_words의 사용법**
`extract_words` , `extract_add_words`의 경우 메소드 내에서 입력된 str을 여러번 순회하는 작업이 수행합니다.
따라서 단순히 str의 iterable를 입력하는 것은 안되며, 이를 list로 변환하거나 `IterableTextFile` 처럼 str의 iterable을 반환하는 객체를 만들어 사용해야 합니다.
```python
# 다음 코드는 test.txt의 내용을 한 번만 순회 가능하기 때문에 오류가 발생합니다.
kiwi.extract_words(open('test.txt'), 10, 10, 0.25)

# list로 변환하면 여러 번 순회가 가능하여 정상적으로 작동합니다.
kiwi.extract_words(list(open('test.txt')), 10, 10, 0.25) 

# test.txt 파일의 크기가 수 GB 이상으로 큰 경우 전체를 메모리에 올리는 것이 비효율적일 수 있습니다.
# 이 경우 다음과 같이 `IterableTextFile`를 정의하여 파일 내용을 여러 번 순회가능한 객체를 사용하면
# 메모리 효율적인 방법으로 extract_words를 사용할 수 있습니다.
class IterableTextFile:
    def __init__(self, path):
        self.path = path

    def __iter__(self):
        yield from open(path, encoding='utf-8')

kiwi.extract_words(IterableTextFile('test.txt'), 10, 10, 0.25)
```

사용자 정의 사전 포맷
---------------------
사용자 정의 사전은 UTF-8로 인코딩된 텍스트 파일이어야 하며, 다음과 같은 구조를 띄어야 합니다.
```text
#으로 시작하는 줄은 주석 처리됩니다.
# 각 필드는 Tab(\t)문자로 구분됩니다.
#
# <단일 형태소를 추가하는 경우>
# (형태) \t (품사태그) \t (점수)
# * (점수)는 생략시 0으로 처리됩니다.
키위	NNP	-5.0

# <이미 존재하는 형태소의 이형태를 추가하는 경우>
# (이형태) \t (원형태소/품사태그) \t (점수)
# * (점수)는 생략시 0으로 처리됩니다.
기위	키위/NNG	-3.0

# <기분석 형태를 추가하는 경우>
# (형태) \t (원형태소/품사태그 + 원형태소/품사태그 + ...) \t (점수)
# * (점수)는 생략시 0으로 처리됩니다.
사겼다	사귀/VV + 었/EP + 다/EF	-1.0
#
# 현재는 공백을 포함하는 다어절 형태를 등록할 수 없습니다.
#
# <규칙 기반의 변형된 이형태를 추가하는 경우>
# (형태 규칙)$ \t (변형된 형태/품사태그) \t (점수)
# 예) 요$	용/EF	-5
```
단어점수는 생략 가능하며, 생략 시 기본값인 0으로 처리됩니다.
실제 예시에 대해서는 Kiwi에 내장된 기본 사전 파일인 https://raw.githubusercontent.com/bab2min/Kiwi/main/models/base/default.dict 을 참조해주세요.

또한 0.14.0버전부터 내장된 기본 오타 사전이 추가되었습니다. 이에 대해서는 https://raw.githubusercontent.com/bab2min/Kiwi/main/models/base/typo.dict 을 참조해주세요.

언어 모델
---------
Kiwi는 최적의 형태소 조합을 탐색하기 위해 내부적으로 언어 모델을 사용합니다. 
0.13.0 버전 이전까지는 Kneser-ney 언어 모델(`knlm`)만을 사용했지만, 0.13.0버전부터 SkipBigram(`sbg`)이라는 새로운 언어 모델에 대한 지원이 추가되었습니다.
기본값은 `knlm`로 설정되어 있지만, 상황에 따라 이용자가 더 적절한 모델을 선택하여 사용할 수 있습니다. 각 모델의 특징은 다음과 같습니다.

* knlm: 0.12.0버전까지 기본적으로 제공되던 모델로, 속도가 빠르고 짧은 거리 내의 형태소(주로 2~3개) 간의 관계를 높은 정확도로 모델링할 수 있습니다.
  그러나 먼 거리의 형태소 간의 관계는 고려하지 못하는 한계가 있습니다.
* sbg: 0.13.0버전에서 추가된 모델로, sbg를 사용시 내부적으로 knlm의 결과에 SkipBigram 결과를 보정하는 식으로 구동됩니다.
  `knlm`에 비해 약 30%정도 처리 시간이 늘어나지면, 먼 거리의 형태소(실질 형태소 기준 최대 8개까지) 간의 관계를 적당한 정확도로 모델링할 수 있습니다.

두 모델 간의 분석 결과 차이는 다음처럼 형태소의 모호성이 먼 거리의 형태소를 통해 해소되는 경우 잘 드러납니다.
```python
>> kiwi = Kiwi(model_type='knlm')
>> kiwi.tokenize('이 번호로 전화를 이따가 꼭 반드시 걸어.')
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

>> kiwi = Kiwi(model_type='sbg')
>> kiwi.tokenize('이 번호로 전화를 이따가 꼭 반드시 걸어.')
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
```

오타 교정
---------
연속적인 문자열을 처리하는 모델의 경우, 특정 지점에서 분석 오류가 발생하면 그 오류 때문에 뒤따르는 분석 결과들이 전부 틀려버리는 경우가 종종 있습니다.
이를 개선하기 위해 0.13.0버전부터 간단한 수준의 오타를 자동으로 교정하는 기능이 추가되었습니다. 
오타 교정을 위해서는 특정 형태소가 어떤 식으로 오타로 변형되는지 정의한, 오타 정의자가 필요합니다. 패키지에는 다음과 같이 세 종류의 기본 오타 정의자가 내장되어 있습니다.

* `kiwipiepy.basic_typos` (`'basic'`): 형태소 내의 오타를 교정하는 기본적인 오타 정의자입니다.
* `kiwipiepy.continual_typos` (`'continual'`): 형태소 간의 연철 오타(`책을` <- `채글`)를 교정하는 오타 정의자입니다. (v0.17.1부터 지원)
* `kiwipiepy.lengthening_typos` (`'lengthening'`): 한 음절을 여러 음절로 늘려 적은 오타(`진짜` <- `지인짜`)를 교정하는 오타 정의자입니다. (v0.19.0부터 지원)
* `kiwipiepy.basic_typos_with_continual` (`'basic_with_continual'`): basic과 continual 두 오타 정의자를 합친 오타 정의자입니다. (v0.17.1부터 지원)
* `kiwipiepy.basic_typos_with_continual_and_lengthening` (`'basic_with_continual_and_lengthening'`): basic, continual, lengthening 세 오타 정의자를 합친 오타 정의자입니다. (v0.19.0부터 지원)

위의 기본 오타 정의자를 사용하거나 혹은 직접 오타 정의자를 정의하여 사용할 수 있습니다.
```python
>>> from kiwipiepy import Kiwi, TypoTransformer, TypoDefinition
# 'basic' 대신 kiwipiepy.basic_typos이라고 입력해도 됨
>>> kiwi = Kiwi(typos='basic')
# 초기에는 로딩 시간으로 5~10초 정도 소요됨
>>> kiwi.tokenize('외않됀대?') 
[Token(form='왜', tag='MAG', start=0, len=1),
 Token(form='안', tag='MAG', start=1, len=1),
 Token(form='되', tag='VV', start=2, len=1),
 Token(form='ᆫ대', tag='EF', start=2, len=2),
 Token(form='?', tag='SF', start=4, len=1)]
# 오타 교정 비용을 변경할 수 있음. 기본값은 6
>>> kiwi.typo_cost_weight = 6 
>>> kiwi.tokenize('일정표를 게시했다')
[Token(form='일정표', tag='NNG', start=0, len=3),
 Token(form='를', tag='JKO', start=3, len=1), 
 Token(form='게시', tag='NNG', start=5, len=2), 
 Token(form='하', tag='XSV', start=7, len=1), 
 Token(form='었', tag='EP', start=7, len=1), 
 Token(form='다', tag='EF', start=8, len=1)]

 # 교정 비용을 낮추면 더 적극적으로 교정을 수행함. 맞는 표현도 과도교정될 수 있으니 주의
>>> kiwi.typo_cost_weight = 2
>>> kiwi.tokenize('일정표를 게시했다')
[Token(form='일정표', tag='NNG', start=0, len=3),
 Token(form='를', tag='JKO', start=3, len=1), 
 Token(form='개시', tag='NNG', start=5, len=2), # '게시'가 맞는 표현이지만 '개시'로 잘못 교정되었음
 Token(form='하', tag='XSV', start=7, len=1), 
 Token(form='었', tag='EP', start=7, len=1), 
 Token(form='다', tag='EF', start=8, len=1)]

# 연철 오타 예제
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

# basic_with_continual 사용 예시
>>> kiwi = Kiwi(typos='basic_with_continual')
>>> kiwi.tokenize('웨 지가캤니?')
[Token(form='왜', tag='MAG', start=0, len=1),
 Token(form='지각', tag='NNG', start=2, len=3),
 Token(form='하', tag='XSV', start=4, len=1),
 Token(form='었', tag='EP', start=4, len=1),
 Token(form='니', tag='EC', start=5, len=1),
 Token(form='?', tag='SF', start=6, len=1)]
```
오타 정의자를 직접 정의하는 방법에 대해서는 `kiwipiepy.TypoTransformer` 를 참조하십시오. 

오타 교정 기능을 사용할 경우 Kiwi 초기화 시에 약 5~10초 정도의 시간이 추가로 소요되며, 문장 당 처리시간은 2배 정도로 늘어납니다. 메모리 사용량은 약 2~3배 정도 증가합니다.

데모
----
https://lab.bab2min.pe.kr/kiwi 에서 데모를 실행해 볼 수 있습니다.

라이센스
--------
Kiwi는 LGPL v3 라이센스로 배포됩니다.

오류 제보
---------
Kiwipiepy 사용 중 오류 발생시 깃헙 이슈탭을 통해 제보해주세요.

Python 모듈 관련 오류는  https://github.com/bab2min/kiwipiepy/issues, 형태소 분석기 전반에 대한 오류는 https://github.com/bab2min/kiwi/issues 에 올려주시면 감사하겠습니다.

태그 목록
---------
세종 품사 태그를 기초로 하되, 일부 품사 태그를 추가/수정하여 사용하고 있습니다.

<style>
.sp{width:100%;}
.sp th, .sp td {border:2px solid #cfd; padding:0.25em 0.5em; }
.sp tr:nth-child(odd) td {background:#f7fffd;}
</style>

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

역사
----
* 0.19.0 (2024-10-03)
    * Kiwi 0.19.0의 기능들(https://github.com/bab2min/Kiwi/releases/tag/v0.19.0 )이 반영되었습니다.
        * 장음화 오타 정정 기능 추가(ex: 지인짜 -> 진짜). Kiwi 초기화 시 typos='lengthening' 옵션으로 사용 가능합니다.
        * 분석 속도 평균 30% 향상
        * 순서 있는 글머리가 여럿 섞인 문장을 분석할 때 종결어미 `-다.`가 종종 SB 태그로 오분석되는 버그 수정
    * `Match.JOIN_*` 옵션으로 생성된 형태소 분석결과를 다시 `Kiwi.join`에 넣을 경우 크래시가 발생하던 버그 수정
    * `TypoTransformer`에 `copy()`, `update()`, `scale_cost()` 메소드 및 `|` 연산자, `*` 연산자 오버로딩이 추가되었습니다.
    * Python>=3.9용 패키지에 대해 numpy 2와 호환성을 갖췄습니다.

* 0.18.1 (2024-09-08)
    * Kiwi 0.18.1의 기능들(https://github.com/bab2min/Kiwi/releases/tag/v0.18.1 )이 반영되었습니다.
        * PreTokenizedSpan과 SPLIT_COMPLEX를 동시에 사용시 종종 빈 결과값이 나오던 버그가 수정되었습니다.
        * 공백 없이 길게 이어진 텍스트를 분석할때 종종 std::length_error가 발생하던 오류가 수정되었습니다.
        * 문장 분리 시 여는 따옴표가 종종 잘못된 문장에 붙던 버그가 수정되었습니다.
    * `Kiwi.tokenize()`에 `compatible_jamo` 인자가 추가되었습니다. compatible_jamo를 True로 설정하면 첫가끝 자모를 호환용 자모로 변환하여 출력합니다.

* 0.18.0 (2024-07-07)
    * Kiwi 0.18.0의 기능들(https://github.com/bab2min/Kiwi/releases/tag/v0.18.0 )이 반영되었습니다.
        * 이모지를 분리하는 `Match.EMOJI` 옵션과 이모지에 해당하는 태그인 `W_EMOJI`가 추가되었습니다.
        * 외국어 및 특수 기호 태그(`SL`, `SH`, `SW`, `W_EMOJI`)에 대해 해당 문자가 속한 언어 집합을 나타내는 `script` 필드가 추가되었습니다. 전체 script의 목록은 `Kiwi.list_all_scripts()` 메소드를 통해 확인할 수 있습니다.
        * 이제 라틴 문자 사이에 악센트가 붙은 문자가 섞여 있는 경우에도 전체 단어가 하나의 형태소로 분석됩니다.
    * `KiwiTokenizer`가 `transformers>=4.41`에서 작동하지 않는 버그가 수정되었습니다.

* 0.17.1 (2024-04-13)
    * Kiwi 0.17.1의 기능들(https://github.com/bab2min/Kiwi/releases/tag/v0.17.1 )이 반영되었습니다.
        * 연철 오타를 교정하는 기능이 추가되었습니다.
        * 문장 분리 정확도가 향상되었습니다.

* 0.17.0 (2024-03-10)
    * Kiwi 0.17.0의 기능들(https://github.com/bab2min/Kiwi/releases/tag/v0.17.0 )이 반영되었습니다.
        * 공백이 포함된 단어를 사전에 등록할 수 있도록 개선되었습니다.
        * 기본 다어절 명사 사전이 추가되었습니다. `Kiwi.__init__()`의 `load_multi_dict` 인자를 통해 기본 다어절 명사 사전의 로드 유무를 설정할 수 있습니다.
        * 공백이 없는 긴 문자열을 분석할 때 크래시가 발생하거나 속도가 느려지는 버그를 수정했습니다.
    * `Kiwi.join()`에 `return_positions` 인자가 추가되었습니다. 이 인자를 통해 각 형태소들의 결합 후 위치를 구할 수 있습니다.
    * `Kiwi.load_user_dictionary()`를 비롯한 일부 메소드에서 잘못된 값이 입력된 경우 크래시가 발생하던 버그가 수정되었습니다.

* 0.16.2 (2023-11-20)
    * `Stopwords`와 `blocklist`를 동시에 사용할 때 종종 크래시가 발생하던 문제가 수정되었습니다.

* 0.16.1 (2023-11-04)
    * Kiwi 0.16.1의 기능들(https://github.com/bab2min/Kiwi/releases/tag/v0.16.1 )이 반영되었습니다.
        * `-다.` 형태로 끝나는 문장어미가 SB로 과도하게 오분석되던 버그가 수정되었습니다.
    * 한국어 템플릿을 위한 편의 기능인 `Kiwi.template`이 추가되었습니다.

* 0.16.0 (2023-08-31)
    * Kiwi 0.16.0의 기능들(https://github.com/bab2min/Kiwi/releases/tag/v0.16.0 )이 반영되었습니다.
        * PretokenizedSpan과 관련된 기능 추가
        * 순서 있는 글머리 기호를 나타내는 SB 태그 추가. `가.`, `나.`, `다.` 등의 글머리 기호가 별도의 문장으로 분리되지 않도록 개선
        * 사용자지정 태그로 사용할 수 있는 USER0 ~ USER4 태그 추가
    * 정규표현식 기반으로 형태소를 사전에 추가하는 `Kiwi.add_re_word` 메소드 추가
    * `Token.span` 추가
    * `Token.user_value` 추가 및 user_value를 설정할 수 있도록 `Kiwi.add_user_word` 계열의 메소드에 `user_value` 인자 추가
    * deprecated 되었던 메소드들 제거
    * `Kiwi.add_pre_analyzed_word`에서 시작위치/끝위치를 지정하지 않았지만 그 값이 자명한 경우, 자동으로 채워넣는 기능 추가
    * `Kiwi.split_into_sents`에 `stopwords` 인자 추가

* 0.15.2 (2023-06-14)
    * Kiwi 0.15.2의 기능들(https://github.com/bab2min/Kiwi/releases/tag/v0.15.2 )이 반영되었습니다.
        * 매우 긴 텍스트를 분석할 때 시작 지점이 잘못 나오는 버그 수정
        * U+10000 이상의 문자가 여러 개 포함된 텍스트를 SwTokenizer로 encode할때 offset이 누락되는 버그 수정
    * `Kiwi.join`에서 형태소 결합 시 띄어쓰기 유무를 설정할 수 있는 기능 추가
    * `Kiwi.tokenize`로 형태소 분석 후 다시 `Kiwi.join`을 수행하는 경우 원본 텍스트의 띄어쓰기를 최대한 반영하여 결합하도록 개선

* 0.15.1 (2023-05-07)
    * Kiwi 0.15.1의 기능들(https://github.com/bab2min/Kiwi/releases/tag/v0.15.1 )이 반영되었습니다.
    * Subword Tokenizer를 제공하는 모듈인 `kiwipiepiy.sw_tokenizer`이 추가되었습니다.
    * huggingface의 tokenizer와 호환가능한 Subword Tokenizer를 제공하는 모듈인 `kiwipiepy.transformers_addon`이 추가되었습니다.

* 0.15.0 (2023-03-23)
    * Kiwi 0.15.0의 기능들(https://github.com/bab2min/Kiwi/releases/tag/v0.15.0 )이 반영되었습니다.
        * 둘 이상의 형태소로 더 잘게 분리될 수 있는 형태소를 추가 분리하는 옵션인 `splitComplex` 도입
        * 부사파생접사를 위한 `XSM` 태그 추가 및 이에 해당하는 형태소 `-이`, `-히`, `-로`, `-스레` 추가
        * 조사/어미에 덧붙는 받침을 위한 `Z_CODA` 태그 추가 및 조사/어미에서 자동으로 Z_CODA를 분절해내는 기능 추가
        * 형태 분석 및 언어 모델 탐색 속도 최적화
        * 옛한글 문자를 특수 기호로 분리하지 않고 일반 한글과 동일하게 처리하도록 개선
        * 형태소 분석 기반의 Subword Tokenizer 구현 (현재 실험적으로 지원 중)
        * 문장 분리 성능 개선
            * `2010. 01. 01.` 와 같이 공백이 포함된 serial 패턴 처리 보강
            * `Dr., Mr.` 와 같이 약자 표현의 `.`이 마침표로 처리되지 않도록 보강
            * '-음'으로 문장이 끝나는 경우를 판별하기 위해 `음/EF` 형태소 추가 및 모델 보강
        * 한 문장 내에서 사전에 미등재된 형태가 256개 이상 등장할 때 형태소 분석 결과가 잘못 나오는 문제 해결
        * 특정 경우에 문장 분리가 전혀 이뤄지지 않던 버그 수정
        * 이모지 등 U+10000 이상의 유니코드 문자를 모두 한자로 분류하던 버그 수정
    * `Kiwi.glue` 에 `insert_new_lines` 인자가 추가되었습니다.
    * 형태소의 사전 표제형을 보여주는 `Token.lemma` 프로퍼티가 추가되었습니다.

* 0.14.1 (2022-12-24)
    * Kiwi 0.14.1의 기능들(https://github.com/bab2min/Kiwi/releases/tag/v0.14.1 )이 반영되었습니다.
        * 특정 텍스트에 대해 형태소 분할 없이 전체 결과를 그대로 반환하는 오류 해결
        * EF 뒤에 보조용언이 따라오는 경우 문장을 분리하지 않도록 개선
    * 이제 Python 3.11을 지원합니다.
        * 추가로 이제 macOS용 binary wheel을 arm64, x86_64로 나누어서 제공합니다.

* 0.14.0 (2022-09-01)
    * Kiwi 0.14.0의 기능들(https://github.com/bab2min/Kiwi/releases/tag/v0.14.0 )이 반영되었습니다.
        * 동사 '이르다'의 모호성 해소 기능 추가
        * W_SERIAL 태그 추가. SS 태그를 SSO, SSC 태그로 세분화
        * 인용문 등으로 둘러싸인 안긴 문장이 포함된 문장에 대해 문장 분리 성능 개선
        * `랬/댔/잖`의 분석 정확도 개선
        * 내장 오타 사전 추가. 사용을 원치 않는 경우 `Kiwi(load_typo_dict=False)`로 끌 수 있습니다.
    * 각종 버그가 수정되었습니다.
        * 오타 교정 기능이 켜져 있는 경우 `Kiwi.join`이 실패하는 문제 해결
        * 사용자 사전에 숫자를 포함한 NNP를 추가해도 반영이 되지 않는 문제 해결
        * `Kiwi.join`이 일부 텍스트를 잘못 결합시키는 오류 해결

* 0.13.1 (2022-07-05)
    * Kiwi 0.13.1의 기능들(https://github.com/bab2min/Kiwi/releases/tag/v0.13.1 )이 반영되었습니다.
        * `Kiwi.join` 이 일부 입력에 대해 오류를 발생시키는 문제를 해결했습니다.

* 0.13.0 (2022-06-28)
    * Kiwi 0.13.0의 기능들(https://github.com/bab2min/Kiwi/releases/tag/v0.13.0 )이 반영되었습니다.
        * 형태소 분석 시 간단한 오타 교정을 수행하는 기능 추가
        * SkipBigram 언어 모델 추가. `Kiwi(model_type='sbg')` 로 사용 가능
        * 분석 결과에서 개별 형태소의 오타 교정 비용을 반환하는 `Token.typo_cost` 필드, 오타 교정 전 형태를 반환하는 `Token.raw_form` 필드 추가
    * 각종 버그가 수정되었습니다.
        * 배포 판에서 `stopwords.txt` 파일이 누락되었던 버그 수정

* 0.12.0 (2022-05-10)
    * Kiwi 0.12.0의 기능들(https://github.com/bab2min/Kiwi/releases/tag/v0.12.0 )이 반영되었습니다.
        * 형태소에 불규칙 활용 여부를 반영하는 `Token.regularity` 필드 추가
        * 분석 결과에서 개별 형태소의 언어 모델 점수를 반영하는 `Token.score` 필드 추가
        * 동사 '걷다'와 '묻다'의 모호성 해소 기능 추가
        * 형태소 결합 기능을 수행하는 `Kiwi.join` 메소드 추가
    * 각종 버그가 수정되었습니다.
        * 특정 상황에서 소수점 패턴이 숫자 - 마침표 - 숫자로 오분석되는 버그
        * 문장 분리 시 종결어미 - 조사로 이어지는 한 문장이 두 문장으로 분리되는 버그
        * `있소`, `잇따라`, `하셔` 등의 표현이 제대로 분석되지 않는 버그

* 0.11.2 (2022-04-14)
    * Kiwi 0.11.2의 기능들(https://github.com/bab2min/Kiwi/releases/tag/v0.11.2 )이 반영되었습니다.
        * 특수 문자가 섞인 텍스트 중 일부가 잘못 분석되는 버그가 수정되었습니다.
        * 특정한 패턴의 텍스트를 입력할 경우 분석 결과가 빈 값으로 나오는 버그가 수정되었습니다.
        * 받침 정규화 기능(normalizeCoda)이 모든 받침에 대해 적용되었습니다.
    * `Kiwi.tokenize`에 `echo` 인자가 추가되었습니다.

* 0.11.1 (2022-04-03)
    * Kiwi 0.11.1의 기능들(https://github.com/bab2min/Kiwi/releases/tag/v0.11.1 )이 반영되었습니다.
        * Windows 환경에서 한글이나 기타 유니코드를 포함한 경로에 위치한 모델을 읽지 못하는 버그가 수정되었습니다.
        * 이제 소수점, 자리 구분 쉼표가 섞인 숫자도 SN 품사태그로 제대로 분석됩니다.
        * `Kiwi.space_tolerance`, `Kiwi.space_penalty` 프로퍼티가 추가되었습니다.
    * 여러 줄의 텍스트를 결합할 때 공백을 적절히 삽입해주는 메소드인 `Kiwi.glue`, 띄어쓰기 교정을 실시하는 메소드인 `Kiwi.space`가 추가되었습니다.

* 0.11.0 (2022-03-19)
    * Kiwi 0.11.0의 기능들(https://github.com/bab2min/Kiwi/releases/tag/v0.11.0 )이 반영되었습니다.
        * 이용자 사전을 관리하는 메소드 `Kiwi.add_pre_analyzed_word`, `Kiwi.add_rule`, `Kiwi.add_re_rule`가 추가되었습니다.
        * 분석 시 접두사/접미사 및 동/형용사 파생접미사의 분리여부를 선택할 수 있는 옵션 `Match.JOIN_NOUN_PREFIX`, `Match.JOIN_NOUN_SUFFIX`, `Match.JOIN_VERB_SUFFIX`, `Match.JOIN_ADJ_SUFFIX`가 추가되었습니다.
        * 결합된 형태소 `Token`의 `start`, `end`, `length`가 부정확한 버그를 수정했습니다.
        * 이제 형태소 결합 규칙이 Kiwi 모델 내로 통합되어 `Kiwi.add_user_word`로 추가된 동/형용사의 활용형도 정상적으로 분석이 됩니다.
        * 언어 모델의 압축 알고리즘을 개선하여 초기 로딩 속도를 높였습니다.
        * SIMD 최적화가 개선되었습니다.
        * 언어 모델 및 기본 사전을 업데이트하여 전반적인 정확도를 높였습니다.
        * 문장 분리 기능의 정확도가 향상되었습니다.

* 0.10.3 (2021-12-22)
    * Kiwi 0.10.3의 기능들(https://github.com/bab2min/Kiwi/releases/tag/v0.10.3 )이 반영되었습니다.
        * `Token`에 `sent_position`, `line_number` 프로퍼티가 추가되었습니다.
        * `Kiwi.split_into_sents` 메소드가 추가되었습니다.
        * SIMD 최적화가 강화되었습니다.
    * pip를 통해 소스코드 설치가 잘 작동하지 않던 문제가 해결되었습니다.
    * `Kiwi.tokenize` 메소드에 stopwords 인자가 추가되었습니다.
    * `kiwipiepy.utils.Stopwords` 에 불용 태그 기능이 추가되었습니다.

* 0.10.2 (2021-11-12)
    * Kiwi 0.10.2의 기능들(https://github.com/bab2min/Kiwi/releases/tag/v0.10.2 )이 반영되었습니다.
        * `Token` 에 `word_position` 프로퍼티가 추가되었습니다.
        * `Kiwi.analyze` 에 `normalize_coda` 인자가 추가되었습니다.
    * `Kiwi.tokenize` 메소드가 추가되었습니다. `analyze` 메소드와는 다르게 바로 분서결과인 `Token`의 `list`를 반환하므로 더 간편하게 사용할 수 있습니다.
    * 불용어 관리 기능을 제공하는 `kiwipiepy.utils.Stopwords` 클래스가 추가되었습니다.

* 0.10.1 (2021-09-06)
    * macOS에서 pip를 통한 설치가 제대로 지원되지 않던 문제를 해결했습니다.
    * `load_user_dictionary` 사용시 품사 태그 뒤에 공백문자가 뒤따르는 경우 태그 해석에 실패하는 문제를 해결했습니다.

* 0.10.0 (2021-08-15)
    * API를 Python에 걸맞게 개편하였습니다. 일부 불편한 메소드들은 사용법이 변경되거나 삭제되었습니다. 이에 대해서는 `0.10.0 버전 변경사항` 단락을 확인해주세요.
    * `prepare` 없이 `analyze` 를 호출할 때 크래시가 발생하던 문제를 수정했습니다.
    * Linux 환경에서 `extract_words` 를 호출할 때 크래시가 발생하던 문제를 수정했습니다.
    * Linux 환경에서 `Options.INTEGRATE_ALLOMORPH` 를 사용시 크래시가 발생하던 문제를 수정했습니다.
    * 이제 형태소 분석 결과가 `tuple` 이 아니라 `Token` 타입으로 반환됩니다. 
    * 형태소 분석 모델 포맷이 최적화되어 파일 크기가 약 20% 작아졌습니다.

* 0.9.3 (2021-06-06)
    * Linux 환경에서 특정 단어가 포함된 텍스트를 분석할 때 크래시가 발생하던 문제를 수정했습니다.
    
* 0.9.2 (2020-12-03)
    * 0.9.1에서 제대로 수정되지 않은 mimalloc 충돌 문제를 수정했습니다.
    * 형태소 분석 모델을 분리하여 패키징하는 기능을 추가했습니다. 용량 문제로 업로드 못했던 대용량 모델을 차차 추가해나갈 예정입니다.

* 0.9.1 (2020-12-03)
    * kiwipiepy가 다른 Python 패키지와 함께 사용될 경우 종종 mimalloc이 충돌하는 문제를 해결했습니다.

* 0.9.0 (2020-11-26)
    * analyze 메소드에서 오류 발생시 exception 발생대신 프로그램이 죽는 문제를 해결했습니다.
    * `default.dict` 에 포함된 활용형 단어 때문에 발생하는 오분석을 수정했습니다.
    * 멀티스레딩 사용시 발생하는 메모리 누수 문제를 해결했습니다.
    * 형태소 탐색 시 조사/어미의 결합조건을 미리 고려하도록 변경하여 속도가 개선되었습니다.
    * 일부 명사(`전랑` 처럼 받침 + 랑으로 끝나는 사전 미등재 명사) 입력시 분석이 실패하는 버그를 수정했습니다.
    * 공백문자만 포함된 문자열 입력시 분석결과가 `/UN` 로 잘못나오는 문제를 수정했습니다.

* 0.8.2 (2020-10-13)
    * W_URL, W_EMAIL, W_HASHTAG 일치 이후 일반 텍스트가 잘못 분석되는 오류를 수정했습니다.
    * W_MENTION을 추가했습니다.
    * 특정 상황에서 결합조건이 무시되던 문제를 해결했습니다. (ex: `고기를 굽다 -> 고기/NNG + 를/JKO + 굽/VV + 이/VCP + 다/EF + ./SF` )

* 0.8.1 (2020-04-01)
    * U+10000 이상의 유니코드 문자를 입력시 Python 모듈에서 오류가 발생하는 문제를 수정했습니다.

* 0.8.0 (2020-03-29)
    * URL, 이메일, 해시태그를 검출하는 기능이 추가되었습니다. `analyze` 메소드의 `match_options` 파라미터로 이 기능의 사용 유무를 설정할 수 있습니다.
    * 치(하지), 컨대(하건대), 토록(하도록), 케(하게) 축약형이 포함된 동사 활용형을 제대로 분석하지 못하는 문제를 해결했습니다.
    * 사용자 사전에 알파벳이나 숫자, 특수 기호가 포함된 단어가 있을 때, 형태소 분석시 알파벳, 숫자, 특수 기호가 포함된 문장이 제대로 분석되지 않는 문제를 수정했습니다.
    * 사용자 사전에 형태는 같으나 품사가 다른 단어를 등록할 수 없는 제한을 해제하였습니다.

* 0.7.6 (2020-03-24)
    * `async_analyze` 메소드가 추가되었습니다. 이 메소드는 형태소 분석을 비동기로 처리합니다. 처리 결과는 callable인 리턴값을 호출하여 얻을 수 있습니다.
    * U+10000 이상의 유니코드 문자에 대해 형태소 분석 결과의 위치 및 길이가 부정확하게 나오는 문제를 해결했습니다.

* 0.7.5 (2020-03-04)
    * U+10000 이상의 문자를 입력시 extract 계열 함수에서 종종 오류가 발생하던 문제를 해결했습니다.
    * gcc 4.8 환경 및 manylinux 대한 지원을 추가했습니다.

* 0.7.4 (2019-12-30)
    * reader, receiver를 사용하는 함수 계열에서 메모리 누수가 발생하던 문제를 해결했습니다.
    * 문서 내 reader, receiver의 사용법 내의 오류를 적절하게 수정했습니다.
    * 종종 분석 결과에서 빈 /UN 태그가 등장하는 문제를 수정했습니다.
    * 일부 특수문자를 분석하는데 실패하는 오류를 수정했습니다.

* 0.7.3 (2019-12-15)
    * macOS 환경에서 extract 계열 함수를 호출할때 스레드 관련 오류가 발생하는 문제를 해결했습니다.

* 0.7.2 (2019-12-01)

* 0.7.1 (2019-09-23)
    * 사전 로딩 속도를 개선했습니다.
    * 음운론적 이형태 통합여부를 선택할 수 있도록 옵션을 추가했습니다.

* 0.6.5 (2019-06-22)

* 0.6.4 (2019-06-09)

* 0.6.3 (2019-04-14)
    * 예외를 좀 더 정교하게 잡아내도록 수정했습니다.
    * 형태소 분석을 바로 테스트해볼 수 있도록 모듈에 대화형 인터페이스를 추가했습니다.

* 0.6.1 (2019-03-26)

* 0.6.0 (2018-12-04)
    * 형태소 검색 알고리즘 최적화로 분석 속도가 향상되었습니다.
    * 전반적인 정확도가 상승되었습니다.

* 0.5.4 (2018-10-11)

* 0.5.2 (2018-09-29)

* 0.5.0 (2018-09-16)
    * Python 모듈 지원이 추가되었습니다.
