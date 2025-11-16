# 문장 분리 성능 평가

이 폴더에는 문장 분리 성능을 평가하기 위한 코드와 데이터가 있습니다. 평가 데이터는 testset 폴더 안에 있으며 다음과 같이 구성되어 있습니다.

* sample: http://semantics.kr/%ED%95%9C%EA%B5%AD%EC%96%B4-%ED%98%95%ED%83%9C%EC%86%8C-%EB%B6%84%EC%84%9D%EA%B8%B0-%EB%B3%84-%EB%AC%B8%EC%9E%A5-%EB%B6%84%EB%A6%AC-%EC%84%B1%EB%8A%A5%EB%B9%84%EA%B5%90/ 에서 소개된 블로그 예제
* blogs, blogs_ko: 다양한 블로그에서 수집한 텍스트
* etn: 명사형 전성 어미를 종결 어미처럼 사용하는 텍스트
* nested: 문장 내 안긴 문장이 있는 텍스트
* tweets: 무작위로 선택된 트윗 텍스트
* v_ending: 사투리를 비롯한 다양한 종결어미를 포함한 텍스트
* wikipedia: 위키백과 문서

또한 다음 네 가지 평가 코드를 제공합니다.

* sentence_split.py: 베이스라인과 `Kiwi.split_into_sents` 의 성능을 평가합니다.
* kss_ref.py: [KSS](https://github.com/hyunwoongko/kss)의 성능을 평가합니다.
* koalanlp_ref.py: [KoalaNLP Python](https://github.com/koalanlp/python-support)에 포함된 다양한 문장 분리기의 성능을 평가합니다.
* bareun_ref.py: [Bareun](https://docs.bareun.ai/)의 성능을 평가합니다.


## 평가 결과 요약

다음은 위의 데이터와 코드를 이용해 평가한 결과를 정리한 것입니다. 
베이스라인은 . ! ? 문자 뒤에 바로 공백이 있는 경우만 문장 종결로 간주하고 분리합니다. 

![Normalized F1](https://bab2min.github.io/kiwipiepy/images/SentSplit_F1.PNG)

![속도](https://bab2min.github.io/kiwipiepy/images/SentSplit_Speed.PNG)

Kiwi가 비교적 빠르면서도 높은 정확도를 달성하고 있는 것을 확인할 수 있습니다.

평가에 사용된 라이브러리 버전은 다음과 같습니다.

| 라이브러리         | 버전 |
|-------------------|-----------|
| Kiwi              | 0.22.0    |
| KSS               | 3.4.2     |
| Koala(Okt)        | 2.1.4     |
| Koala(Hnn)        | 2.1.4     |
| Koala(Kmr)        | 2.1.4     |
| Koala(Rhino)      | 2.1.5     |
| Koala(Eunjeon)    | 2.1.6     |
| Koala(Arirang)    | 2.1.4     |
| Koala(Kkma)       | 2.1.4     |
| Bareun            | v3.0.rc2  |

## 직접 평가 실행해보기
```console
$ python sentence_split.py testset/*.txt

======== Baseline Splitter ========
[Sentence Split Benchmark] Dataset: testset/blogs.txt
Gold: 170 sents, System: 151 sents, EM: 0.53529, F1: 0.66847, Normalized F1: 0.59884, Latency: 2.84 msec

[Sentence Split Benchmark] Dataset: testset/blogs_ko.txt
Gold: 346 sents, System: 243 sents, EM: 0.43642, F1: 0.55724, Normalized F1: 0.52607, Latency: 0.17 msec

[Sentence Split Benchmark] Dataset: testset/etn.txt
Gold: 39 sents, System: 26 sents, EM: 0.46154, F1: 0.59857, Normalized F1: 0.59857, Latency: 0.03 msec

[Sentence Split Benchmark] Dataset: testset/nested.txt
Gold: 91 sents, System: 104 sents, EM: 0.68132, F1: 0.85438, Normalized F1: 0.75991, Latency: 0.08 msec

[Sentence Split Benchmark] Dataset: testset/sample.txt
Gold: 43 sents, System: 26 sents, EM: 0.34884, F1: 0.52431, Normalized F1: 0.52431, Latency: 0.02 msec

[Sentence Split Benchmark] Dataset: testset/tweets.txt
Gold: 178 sents, System: 140 sents, EM: 0.51124, F1: 0.65446, Normalized F1: 0.61806, Latency: 0.08 msec

[Sentence Split Benchmark] Dataset: testset/v_ending.txt
Gold: 30 sents, System: 6 sents, EM: 0.00000, F1: 0.11359, Normalized F1: 0.11359, Latency: 0.02 msec

[Sentence Split Benchmark] Dataset: testset/wikipedia.txt
Gold: 326 sents, System: 267 sents, EM: 0.66258, F1: 0.76664, Normalized F1: 0.76379, Latency: 0.24 msec

[Overall]
Gold: 1223 sents, System: 963 sents, EM: 0.52657, F1: 0.65406, Normalized F1: 0.62247

======== Kiwi.split_into_sents ========
[Sentence Split Benchmark] Dataset: testset/blogs.txt
Gold: 170 sents, System: 183 sents, EM: 0.78824, F1: 0.93152, Normalized F1: 0.86670, Latency: 55.80 msec

[Sentence Split Benchmark] Dataset: testset/blogs_ko.txt
Gold: 346 sents, System: 356 sents, EM: 0.64740, F1: 0.86648, Normalized F1: 0.80688, Latency: 101.15 msec

[Sentence Split Benchmark] Dataset: testset/etn.txt
Gold: 39 sents, System: 40 sents, EM: 0.76923, F1: 0.91126, Normalized F1: 0.88102, Latency: 8.96 msec

[Sentence Split Benchmark] Dataset: testset/nested.txt
Gold: 91 sents, System: 93 sents, EM: 0.76923, F1: 0.91435, Normalized F1: 0.86387, Latency: 57.47 msec

[Sentence Split Benchmark] Dataset: testset/sample.txt
Gold: 43 sents, System: 41 sents, EM: 0.83721, F1: 0.91470, Normalized F1: 0.91470, Latency: 9.58 msec

[Sentence Split Benchmark] Dataset: testset/tweets.txt
Gold: 178 sents, System: 173 sents, EM: 0.74719, F1: 0.86561, Normalized F1: 0.82174, Latency: 35.11 msec

[Sentence Split Benchmark] Dataset: testset/v_ending.txt
Gold: 30 sents, System: 17 sents, EM: 0.26667, F1: 0.45642, Normalized F1: 0.45642, Latency: 4.76 msec

[Sentence Split Benchmark] Dataset: testset/wikipedia.txt
Gold: 326 sents, System: 330 sents, EM: 0.97853, F1: 0.99427, Normalized F1: 0.98267, Latency: 149.78 msec

[Overall]
Gold: 1223 sents, System: 1233 sents, EM: 0.78005, F1: 0.90608, Normalized F1: 0.86601
```

kss_ref.py의 경우 backend 옵션을 제공합니다. [pecab, mecab, none] 중에 하나를 선택할 수 있습니다. 이에 대한 설명은 [KSS 공식 저장소](https://github.com/hyunwoongko/kss)를 참조하세요.
```console
$ python kss_ref.py testset/*.txt

[Sentence Split Benchmark] Dataset: testset/blogs.txt
Gold: 170 sents, System: 168 sents, EM: 0.87059, F1: 0.92162, Normalized F1: 0.88878, Latency: 8312.93 msec

[Sentence Split Benchmark] Dataset: testset/blogs_ko.txt
Gold: 346 sents, System: 330 sents, EM: 0.82659, F1: 0.90335, Normalized F1: 0.88605, Latency: 20854.73 msec

[Sentence Split Benchmark] Dataset: testset/etn.txt
Gold: 39 sents, System: 38 sents, EM: 0.46154, F1: 0.72488, Normalized F1: 0.63843, Latency: 637.50 msec

[Sentence Split Benchmark] Dataset: testset/nested.txt
Gold: 91 sents, System: 88 sents, EM: 0.86813, F1: 0.93012, Normalized F1: 0.92063, Latency: 6230.18 msec

[Sentence Split Benchmark] Dataset: testset/sample.txt
Gold: 43 sents, System: 40 sents, EM: 0.86047, F1: 0.91394, Normalized F1: 0.91394, Latency: 8644.43 msec

[Sentence Split Benchmark] Dataset: testset/tweets.txt
Gold: 178 sents, System: 159 sents, EM: 0.74157, F1: 0.82720, Normalized F1: 0.80771, Latency: 2766.56 msec

[Sentence Split Benchmark] Dataset: testset/v_ending.txt
Gold: 30 sents, System: 17 sents, EM: 0.36667, F1: 0.48153, Normalized F1: 0.48153, Latency: 448.44 msec

[Sentence Split Benchmark] Dataset: testset/wikipedia.txt
Gold: 326 sents, System: 323 sents, EM: 0.98160, F1: 0.98801, Normalized F1: 0.98801, Latency: 171493.01 msec

```

koalanlp_ref.py 역시 backend 옵션을 제공합니다. [OKT, HNN, KMR, RHINO, EUNJEON, ARIRANG, KKMA] 중 하나를 선택할 수 있으며, OKT, HNN의 경우 두 분석기가 자체적으로 제공하는 문장 분리 기능을 사용하고, 나머지의 경우 각 분석기로 형태소 분석 후 KoalaNLP에서 제공하는 문장 분리 기능을 사용합니다.
```console
$ python koalanlp_ref.py testset/*.txt --backend=OKT

[koalanlp.jip] [INFO] Latest version of kr.bydelta:koalanlp-okt (2.1.4) will be used.
[root] Java gateway started with port number 10667
[root] Callback server will use port number 25334
[koalanlp.jip] JVM initialization procedure is completed.
[Sentence Split Benchmark] Dataset: testset/blogs.txt
Gold: 170 sents, System: 141 sents, EM: 0.53529, F1: 0.66847, Normalized F1: 0.62168, Latency: 66.48 msec

[Sentence Split Benchmark] Dataset: testset/blogs_ko.txt
Gold: 346 sents, System: 224 sents, EM: 0.43642, F1: 0.55724, Normalized F1: 0.55064, Latency: 82.07 msec

[Sentence Split Benchmark] Dataset: testset/etn.txt
Gold: 39 sents, System: 26 sents, EM: 0.46154, F1: 0.59857, Normalized F1: 0.59857, Latency: 22.24 msec

[Sentence Split Benchmark] Dataset: testset/nested.txt
Gold: 91 sents, System: 107 sents, EM: 0.79121, F1: 0.93010, Normalized F1: 0.83832, Latency: 56.54 msec

[Sentence Split Benchmark] Dataset: testset/sample.txt
Gold: 43 sents, System: 27 sents, EM: 0.37209, F1: 0.55109, Normalized F1: 0.55109, Latency: 6.44 msec

[Sentence Split Benchmark] Dataset: testset/tweets.txt
Gold: 178 sents, System: 145 sents, EM: 0.53371, F1: 0.69434, Normalized F1: 0.66198, Latency: 63.41 msec

[Sentence Split Benchmark] Dataset: testset/v_ending.txt
Gold: 30 sents, System: 6 sents, EM: 0.00000, F1: 0.11359, Normalized F1: 0.11359, Latency: 3.23 msec

[Sentence Split Benchmark] Dataset: testset/wikipedia.txt
Gold: 326 sents, System: 264 sents, EM: 0.65951, F1: 0.76639, Normalized F1: 0.76354, Latency: 104.33 msec

```
```console
$ python koalanlp_ref.py testset/*.txt --backend=HNN
[koalanlp.jip] [INFO] Latest version of kr.bydelta:koalanlp-hnn (2.1.4) will be used.
[root] Java gateway started with port number 40435
[root] Callback server will use port number 25334
[koalanlp.jip] JVM initialization procedure is completed.
[Sentence Split Benchmark] Dataset: testset/blogs.txt
Gold: 170 sents, System: 151 sents, EM: 0.54118, F1: 0.69341, Normalized F1: 0.62515, Latency: 81.27 msec

[Sentence Split Benchmark] Dataset: testset/blogs_ko.txt
Gold: 346 sents, System: 241 sents, EM: 0.44220, F1: 0.59185, Normalized F1: 0.57098, Latency: 99.12 msec

[Sentence Split Benchmark] Dataset: testset/etn.txt
Gold: 39 sents, System: 26 sents, EM: 0.46154, F1: 0.59857, Normalized F1: 0.59857, Latency: 21.94 msec

[Sentence Split Benchmark] Dataset: testset/nested.txt
Gold: 91 sents, System: 114 sents, EM: 0.78022, F1: 0.94163, Normalized F1: 0.82031, Latency: 51.72 msec

[Sentence Split Benchmark] Dataset: testset/sample.txt
Gold: 43 sents, System: 27 sents, EM: 0.34884, F1: 0.54681, Normalized F1: 0.54681, Latency: 6.40 msec

[Sentence Split Benchmark] Dataset: testset/tweets.txt
Gold: 178 sents, System: 150 sents, EM: 0.54494, F1: 0.70350, Normalized F1: 0.66922, Latency: 61.29 msec

[Sentence Split Benchmark] Dataset: testset/v_ending.txt
Gold: 30 sents, System: 6 sents, EM: 0.00000, F1: 0.11359, Normalized F1: 0.11359, Latency: 3.42 msec

[Sentence Split Benchmark] Dataset: testset/wikipedia.txt
Gold: 326 sents, System: 328 sents, EM: 0.67791, F1: 0.98116, Normalized F1: 0.97286, Latency: 126.11 msec
```

bareun_ref.py를 실행하면 바른 형태소 분석기의 문장 분리 성능을 평가할 수 있습니다. 이를 위해서는 먼저 [바른 형태소 분석기를 설치](https://docs.bareun.ai/install/overview/)하고 API Key를 받아야 합니다. 이에 대한 자세한 내용은 [바른 형태소 분석기 공식 문서](https://docs.bareun.ai/)를 참조하세요.
```console
$ python bareun_ref.py testset/*.txt --api_key=${YOUR_API_KEY}
[Sentence Split Benchmark] Dataset: E:\CppRepo\kiwipiepy\benchmark\sentence_split\testset\blogs.txt
Gold: 170 sents, System: 112 sents, EM: 0.42941, F1: 0.56816, Normalized F1: 0.56364, Latency: 2325.35 msec

[Sentence Split Benchmark] Dataset: E:\CppRepo\kiwipiepy\benchmark\sentence_split\testset\blogs_ko.txt
Gold: 346 sents, System: 190 sents, EM: 0.36705, F1: 0.46984, Normalized F1: 0.46007, Latency: 742.79 msec

[Sentence Split Benchmark] Dataset: E:\CppRepo\kiwipiepy\benchmark\sentence_split\testset\etn.txt
Gold: 39 sents, System: 24 sents, EM: 0.43590, F1: 0.54637, Normalized F1: 0.54637, Latency: 142.94 msec

[Sentence Split Benchmark] Dataset: E:\CppRepo\kiwipiepy\benchmark\sentence_split\testset\nested.txt
Gold: 91 sents, System: 87 sents, EM: 0.73626, F1: 0.84394, Normalized F1: 0.80394, Latency: 450.57 msec

[Sentence Split Benchmark] Dataset: E:\CppRepo\kiwipiepy\benchmark\sentence_split\testset\sample.txt
Gold: 43 sents, System: 20 sents, EM: 0.18605, F1: 0.36379, Normalized F1: 0.36379, Latency: 68.93 msec

[Sentence Split Benchmark] Dataset: E:\CppRepo\kiwipiepy\benchmark\sentence_split\testset\tweets.txt
Gold: 178 sents, System: 111 sents, EM: 0.39888, F1: 0.54465, Normalized F1: 0.53523, Latency: 487.32 msec

[Sentence Split Benchmark] Dataset: E:\CppRepo\kiwipiepy\benchmark\sentence_split\testset\v_ending.txt
Gold: 30 sents, System: 6 sents, EM: 0.00000, F1: 0.11359, Normalized F1: 0.11359, Latency: 51.78 msec

[Sentence Split Benchmark] Dataset: E:\CppRepo\kiwipiepy\benchmark\sentence_split\testset\wikipedia.txt
Gold: 326 sents, System: 302 sents, EM: 0.87730, F1: 0.91588, Normalized F1: 0.91588, Latency: 999.39 msec
```
