# 문장 분리 성능 평가

이 폴더에는 문장 분리 성능을 평가하기 위한 코드와 데이터가 있습니다. 평가 데이터는 testset 폴더 안에 있으며 다음과 같이 구성되어 있습니다.

* sample: http://semantics.kr/%ED%95%9C%EA%B5%AD%EC%96%B4-%ED%98%95%ED%83%9C%EC%86%8C-%EB%B6%84%EC%84%9D%EA%B8%B0-%EB%B3%84-%EB%AC%B8%EC%9E%A5-%EB%B6%84%EB%A6%AC-%EC%84%B1%EB%8A%A5%EB%B9%84%EA%B5%90/ 에서 소개된 블로그 예제
* blogs: 다양한 블로그에서 수집한 텍스트
* tweets: 무작위로 선택된 트윗 텍스트
* v_ending: 사투리를 비롯한 다양한 종결어미를 포함한 텍스트

또한 다음 세 가지 평가 코드를 제공합니다.

* sentence_split.py: 베이스라인과 `Kiwi.split_into_sents` 의 성능을 평가합니다.
* kss_ref.py: [KSS](https://github.com/hyunwoongko/kss)의 성능을 평가합니다.
* koalanlp_ref.py: [KoalaNLP Python](https://github.com/koalanlp/python-support)에 포함된 다양한 문장 분리기의 성능을 평가합니다.


## 평가 결과 요약

다음은 위의 데이터와 코드를 이용해 평가한 결과를 정리한 것입니다. 
베이스라인은 . ! ? 문자 뒤에 바로 공백이 있는 경우만 문장 종결로 간주하고 분리합니다. 

![EM](https://bab2min.github.io/kiwipiepy/images/SentSplit_EM.PNG)
![F1](https://bab2min.github.io/kiwipiepy/images/SentSplit_F1.PNG)

![속도](https://bab2min.github.io/kiwipiepy/images/SentSplit_Speed.PNG)

Kiwi가 비교적 빠르면서도 높은 정확도를 달성하고 있는 것을 확인할 수 있습니다. 다만 아직 전반적으로 부정확한 상황이라 앞으로 개선의 여지가 많습니다.

평가에 사용된 라이브러리 버전은 다음과 같습니다.

|  Kiwi  |   KSS   | Koala(Okt) | Koala(Hnn) | Koala(Kmr) | Koala(Rhino) | Koala(Eunjeon) | Koala(Arirang) | Koala(Kkma) |
|:------:|:-------:|:----------:|:----------:|:----------:|:----------:|:----------:|:----------:|:----------:|
| 0.11.0 | 3.4.2   | 2.1.4      | 2.1.4      | 2.1.4      | 2.1.5      | 2.1.6      | 2.1.4      | 2.1.4      |

## 직접 평가 실행해보기
```console
$ python sentence_split.py testset/*.txt

======== Baseline Splitter ========
[Sentence Split Benchmark] Dataset: testset/blogs.txt
Gold: 175 sents,  System: 151 sents,  EM: 0.53714,  F1: 0.66050,  Latency: 0.32 msec

[Sentence Split Benchmark] Dataset: testset/sample.txt
Gold: 43 sents,  System: 26 sents,  EM: 0.34884,  F1: 0.52431,  Latency: 0.03 msec

[Sentence Split Benchmark] Dataset: testset/tweets.txt
Gold: 184 sents,  System: 139 sents,  EM: 0.50543,  F1: 0.63493,  Latency: 0.16 msec

[Sentence Split Benchmark] Dataset: testset/v_ending.txt
Gold: 30 sents,  System: 6 sents,  EM: 0.00000,  F1: 0.11359,  Latency: 0.02 msec

======== Kiwi.split_into_sents ========
[Sentence Split Benchmark] Dataset: testset/blogs.txt
Gold: 175 sents,  System: 191 sents,  EM: 0.77143,  F1: 0.89778,  Latency: 163.30 msec

[Sentence Split Benchmark] Dataset: testset/sample.txt
Gold: 43 sents,  System: 43 sents,  EM: 0.88372,  F1: 0.93931,  Latency: 29.11 msec

[Sentence Split Benchmark] Dataset: testset/tweets.txt
Gold: 184 sents,  System: 184 sents,  EM: 0.62500,  F1: 0.84072,  Latency: 88.80 msec

[Sentence Split Benchmark] Dataset: testset/v_ending.txt
Gold: 30 sents,  System: 21 sents,  EM: 0.20000,  F1: 0.50358,  Latency: 19.42 msec

```

kss_ref.py의 경우 backend 옵션을 제공합니다. [pynori, mecab, none] 중에 하나를 선택할 수 있습니다. 이에 대한 설명은 [KSS 공식 저장소](https://github.com/hyunwoongko/kss)를 참조하세요.
```console
$ python kss_ref.py testset/*.txt --backend=none

[Sentence Split Benchmark] Dataset: testset/blogs.txt
Gold: 175 sents,  System: 152 sents,  EM: 0.64000,  F1: 0.78595,  Latency: 3065.09 msec

[Sentence Split Benchmark] Dataset: testset/sample.txt
Gold: 43 sents,  System: 34 sents,  EM: 0.58140,  F1: 0.71413,  Latency: 267.51 msec

[Sentence Split Benchmark] Dataset: testset/tweets.txt
Gold: 184 sents,  System: 113 sents,  EM: 0.36413,  F1: 0.51866,  Latency: 4092.26 msec

[Sentence Split Benchmark] Dataset: testset/v_ending.txt
Gold: 30 sents,  System: 8 sents,  EM: 0.00000,  F1: 0.16575,  Latency: 501.14 msec

```

koalanlp_ref.py 역시 backend 옵션을 제공합니다. [OKT, HNN, KMR, RHINO, EUNJEON, ARIRANG, KKMA] 중 하나를 선택할 수 있으며, OKT, HNN의 경우 두 분석기가 자체적으로 제공하는 문장 분리 기능을 사용하고, 나머지의 경우 각 분석기로 형태소 분석 후 KoalaNLP에서 제공하는 문장 분리 기능을 사용합니다.
```console
$ python koalanlp_ref.py testset/*.txt --backend=OKT

[koalanlp.jip] [INFO] Latest version of kr.bydelta:koalanlp-okt (2.1.4) will be used.
[root] Java gateway started with port number 10667
[root] Callback server will use port number 25334
[koalanlp.jip] JVM initialization procedure is completed.
[Sentence Split Benchmark] Dataset: testset/blogs.txt
Gold: 175 sents,  System: 141 sents,  EM: 0.53714,  F1: 0.66050,  Latency: 53.95 msec

[Sentence Split Benchmark] Dataset: testset/sample.txt
Gold: 43 sents,  System: 27 sents,  EM: 0.37209,  F1: 0.55109,  Latency: 7.19 msec

[Sentence Split Benchmark] Dataset: testset/tweets.txt
Gold: 184 sents,  System: 144 sents,  EM: 0.52717,  F1: 0.67351,  Latency: 56.93 msec

[Sentence Split Benchmark] Dataset: testset/v_ending.txt
Gold: 30 sents,  System: 6 sents,  EM: 0.00000,  F1: 0.11359,  Latency: 9.76 msec

```
