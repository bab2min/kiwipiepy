# 모호성 해소 평가

이 폴더에는 모호성 해소 성능을 평가하기 위한 코드와 데이터가 있습니다. 평가 데이터는 testset 폴더 안에 있으며 다음과 같이 구성되어 있습니다.

* irregular_verbs(불규칙활용): 형태가 동일한 불규칙 활용 동사와 규칙 활용 동사를 구분하는 평가
* verb_vs_adj(동/형용사): 형태가 동일한 동사와 형용사를 구분하는 평가
* nouns(명사): 헷갈리기 쉬운 명사 + 조사 조합을 구분하는 평가
* distant(장거리): 위의 평가셋에서 문장의 길이를 늘려 난이도를 높인 평가

또한 다음 평가 코드를 제공합니다.

* disambiguate.py: `Kiwi` 및 다른 형태소 분석기의 성능을 평가합니다.

## 평가 결과 요약

![모호성 해소 성능](https://bab2min.github.io/kiwipiepy/images/DisambAcc.PNG)

Kiwi가 다른 형태소 분석기에 비해 압도적으로 높은 정확도를 보임을 확인할 수 있습니다.

## 직접 평가 실행해보기
다른 형태소 분석기를 테스트하기 위해서는 konlpy 혹은 khaiii를 설치해야합니다.
특히 Mecab 이용시 split_inflect 기능 패치가 추가된 [konlpy 버전](https://github.com/konlpy/konlpy/commit/d9206305195583c08400cb2237c837cc42df2e65)이 필요합니다.

```console
$ python disambiguate.py testset/*.txt --target=kiwi,kiwi_sbg,komoran,mecab,kkma,hannanum,okt,khaiii --error_output_dir=errors/
Initialize kiwipiepy (0.17.1)
Initialize kiwipiepy (0.17.1)
Initialize Komoran from konlpy (0.6.0)
Initialize Mecab from konlpy (0.6.0)
Initialize Kkma from konlpy (0.6.0)
Initialize Hannanum from konlpy (0.6.0)
Initialize Okt from konlpy (0.6.0)
Initialize khaiii (0.4)
                        kiwi   kiwi_sbg komoran mecab   kkma   hannanum  okt    khaiii
distant.txt             0.581   0.774   0.419   0.548   0.419   -       -       0.484
irregular_verbs.txt     0.821   0.896   0.463   0.463   0.522   0.463   0.463   0.552
nouns.txt               0.891   0.891   0.545   0.600   0.709   0.473   0.527   0.673
verb_vs_adj.txt         0.907   0.907   0.407   0.537   0.463   -       -       0.611
```

`Hannanum`과 `Okt`의 경우 동사와 형용사를 별도로 구분하는 기능이 없어서 `verb_vs_adj`이나 `distant` 평가에서 점수를 매기지 않습니다.
