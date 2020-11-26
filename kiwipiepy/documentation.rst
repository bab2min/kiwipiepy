Kiwipiepy란?
============
Kiwipiepy는 한국어 형태소 분석기인 Kiwi(Korean Intelligent Word Identifier)의 Python 모듈입니다. 
C++로 작성되었고 다른 패키지에 의존성이 없으므로 C++ 컴파일이 가능한 환경이라면 어디에서나 Kiwipiepy를 사용 가능합니다.

현재 Kiwipiepy의 최신 버전은 0.9.0입니다.

.. image:: https://badge.fury.io/py/kiwipiepy.svg

시작하기
--------
pip를 이용해 쉽게 설치할 수 있습니다. (https://pypi.org/project/kiwipiepy/)

::

    $ pip install kiwipiepy

지원하는 OS와 Python 버전은 다음과 같습니다:

* Python 3.5 이상이 설치된 Linux (x86-64) 
* Python 3.5 이상이 설치된 macOS 10.13이나 그 이후 버전
* Python 3.5 이상이 설치된 Windows 7 이나 그 이후 버전 (x86, x86-64)
* Python 3.5 이상이 설치된 다른 OS: 이 경우 소스 코드 컴파일을 위해 C++11이 지원되는 컴파일러가 필요합니다.

Kiwipiepy가 제대로 설치되었는지 확인하기 위해서는 다음 명령어를 실행해보십시오.

::

    $ python -m kiwipiepy

위 명령어는 대화형 인터페이스를 시작합니다. 인터페이스에 원하는 문장을 입력하면 형태소 분석 결과를 확인할 수 있습니다.

::

    >> 안녕?
    [('안녕', 'IC', 0, 2), ('?', 'SF', 2, 3)]

인터페이스를 종료하려면 Ctrl + C 를 누르십시오.

예제
----
** 간단한 분석 **

다음 예제 코드는 kiwipiepy 인스턴스를 생성해 형태소 분석을 수행하는 간단한 예제 코드입니다.

::

    from kiwipiepy import Kiwi
    kiwi = Kiwi()
    kiwi.prepare()
    for result, score in kiwi.analyze("형태소 분석 결과입니다", top_n=5):
        print(score, result, sep='\t')
    
    # 위 코드를 실행하면 다음과 같은 결과가 나옵니다.
    # -34.40869140625	[('형태소', 'NNG', 0, 3), ('분석', 'NNG', 4, 2), ('결과', 'NNG', 7, 2), ('이', 'VCP', 9, 1), ('ᆸ니다', 'EF', 10, 2)]
    # -41.41796875	[('형태소', 'NNG', 0, 3), ('분석', 'NNG', 4, 2), ('결과', 'NNG', 7, 2), ('이', 'MM', 9, 1), ('ᆸ니다', 'EF', 10, 2)]
    # -54.265869140625	[('형태소', 'NNG', 0, 3), ('분석', 'NNG', 4, 2), ('결과', 'NNG', 7, 2), ('입', 'NNG', 9, 1), ('니', 'EC', 10, 1), ('다', 'EC', 11, 1)]
    # -54.470458984375	[('형태소', 'NNG', 0, 3), ('분석', 'NNG', 4, 2), ('결과', 'NNG', 7, 2), ('입', 'NNG', 9, 1), ('니다', 'EF', 10, 2)]
    # -54.705078125	[('형태소', 'NNG', 0, 3), ('분석', 'NNG', 4, 2), ('결과', 'NNG', 7, 2), ('입', 'NNG', 9, 1), ('이', 'VCP', 10, 1), ('니', 'EC', 10, 1), ('다', 'EC', 11, 1)]

** 사용자 단어 추가 **

사용자 정의 단어를 추가하여 형태소 분석을 수행하는 예제입니다. 사용자 정의 단어를 등록하면 이는 Kiwi 분석기의 사전에 포함되어 결과의 후보로 등장할 수 있게 됩니다.

종종 동일한 형태의 단어가 여러 가지로 분석되는 경우가 있습니다. 이 경우 사용자 정의 단어를 우선할지, 분석기가 가지고 있는 형태소 정보를 우선할지 사용자 단어 점수를 조절함으로써 통제 가능합니다.
아래 예제는 '골리'라는 고유 명사 단어가 포함된 문장을 분석하는 경우에 부여하는 단어 점수에 따라 결과가 어떻게 달라지는지를 보여줍니다.

::

    from kiwipiepy import Kiwi

    # 사용자 단어 추가 없이 분석해보겠습니다.

    kiwi = Kiwi()
    kiwi.prepare()

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
    kiwi.prepare()

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
    kiwi.prepare()

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

** 멀티스레딩 analyze **

다음 예제 코드는 멀티스레드를 활용하여 `test.txt` 파일을 줄별로 읽어들여 형태소 분석한 뒤 그 결과를 `result.txt`에 저장합니다.

::

    from kiwipiepy import Kiwi
    # 4개의 스레드에서 동시에 처리합니다.
    # num_workers 생략시 현재 환경에서 사용가능한 모든 코어를 다 사용합니다.
    kiwi = Kiwi(num_workers=4)
    kiwi.load_user_dictionary('userDict.txt')
    kiwi.prepare()
    with open('result.txt', 'w', encoding='utf-8') as output:
        for res in kiwi.analyze(open('test.txt', encoding='utf-8')):
            output.write(' '.join(map(lambda x:x[0]+'/'+x[1], res[0][0])) + '\n')

Kiwi() 생성시 인자로 준 num_workers에 따라 여러 개의 스레드에서 작업이 동시에 처리됩니다. 반환되는 값은 입력되는 값의 순서와 동일합니다.

`analyze`를 인자를 str의 iterable로 준 경우 이 iterable을 읽어들이는 시점은 analyze 호출 이후일 수도 있습니다. 
따라서 이 인자가 다른 IO 자원(파일 입출력 등)과 연동되어 있다면 모든 분석이 끝나기 전까지 해당 자원을 종료하면 안됩니다.
예를 들어 다음과 같이 open을 통해 생성한 파일 입출력 객체를 미리 종료하는 경우 오류가 발생할 수 있습니다.

::

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

아래 코드는 0.9.0버전 이전에서 사용되던 멀티스레딩 코드입니다. 현재는 추천되지 않습니다.
아래의 기능은 0.10.0버전부터 제거될 예정입니다.

::

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

** async_analyze 예제 **

다음 예제 코드에서는 async_analyze를 사용해 멀티스레딩 분석을 진행합니다.
async_analyze 메소드 역시 0.10.0버전부터 제거될 예정입니다. 위의 analyze 메소드를 사용하길 권장합니다.

::

    from kiwipiepy import Kiwi
    kiwi = Kiwi()
    kiwi.prepare()
    ret = []
    # input.txt 파일의 라인별로 분석 작업을 할당합니다.
    for line in open('input.txt', encoding='utf-8'):
        ret.append(kiwi.async_analyze(line))

    for r in ret:
        print(r()) # r을 호출하여 분석 결과를 얻습니다.

사용자 정의 사전 포맷
---------------------
사용자 정의 사전은 UTF-8로 인코딩된 텍스트 파일이어야 하며, 다음과 같은 구조를 띄어야 합니다.


    #주석은 #으로 시작합니다.

    단어1 [탭문자] 품사태그 [탭문자] 단어점수

    단어2 [탭문자] 품사태그 [탭문자] 단어점수

    단어3 [탭문자] 품사태그 [탭문자] 단어점수

단어점수는 생략 가능하며, 생략 시 기본값인 0으로 처리됩니다.

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

역사
----
* 0.9.0 (2020-11-26)
    * analyze 메소드에서 오류 발생시 exception 발생대신 프로그램이 죽는 문제를 해결했습니다.
    * `default.dict`에 포함된 활용형 단어 때문에 발생하는 오분석을 수정했습니다.
    * 멀티스레딩 사용시 발생하는 메모리 누수 문제를 해결했습니다.
    * 형태소 탐색 시 조사/어미의 결합조건을 미리 고려하도록 변경하여 속도가 개선되었습니다.
    * 일부 명사(`전랑`처럼 받침 + 랑으로 끝나는 사전 미등재 명사) 입력시 분석이 실패하는 버그를 수정했습니다.
    * 공백문자만 포함된 문자열 입력시 분석결과가 `/UN`로 잘못나오는 문제를 수정했습니다.

* 0.8.2 (2020-10-13)
    * W_URL, W_EMAIL, W_HASHTAG 일치 이후 일반 텍스트가 잘못 분석되는 오류를 수정했습니다.
    * W_MENTION을 추가했습니다.
    * 특정 상황에서 결합조건이 무시되던 문제를 해결했습니다. (ex: `고기를 굽다 -> 고기/NNG + 를/JKO + 굽/VV + 이/VCP + 다/EF + ./SF`)

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
