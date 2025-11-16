# 형태소 분석 평가

이 폴더에는 형태소 분석 성능을 평가하기 위한 코드가 있습니다. 평가 데이터는 https://github.com/bab2min/Kiwi/tree/main/eval_data 에서 구할 수 있습니다.

## 방언 형태소 분석 성능 요약
![방언별 형태소 분석 정확도](https://bab2min.github.io/kiwipiepy/images/DialectAcc.png)

대체로 표준어만 지원하는 다른 형태소 분석기들과 달리, Kiwi는 여러 방언에 대해서도 높은 정확도를 보임을 확인할 수 있습니다. 특히 enabled_dialects 옵션을 통해 방언별 사전을 활성화하면 더욱 높은 성능을 얻을 수 있습니다.

## 직접 평가 실행해보기
다른 형태소 분석기를 테스트하기 위해서는 konlpy 혹은 khaiii를 설치해야합니다.
특히 Mecab 이용시 split_inflect 기능 패치가 추가된 [konlpy 버전](https://github.com/konlpy/konlpy/commit/d9206305195583c08400cb2237c837cc42df2e65)이 필요합니다.
먼저 https://github.com/bab2min/Kiwi/tree/main/eval_data 에서 평가 데이터를 다운 받아서 dataset 폴더에 넣었다고 가정하겠습니다.

```console
$ python tokenize.py dataset/dialect/*.txt --target=kiwi,komoran,mecab,kkma,hannanum,okt,khaiii --kiwi-enabled-dialects all --error_output_dir=errors/
Initialize kiwipiepy (0.22.0)
Initialize Komoran from konlpy (0.6.0)
Initialize Mecab from konlpy (0.6.0)
Initialize Kkma from konlpy (0.6.0)
Initialize Hannanum from konlpy (0.6.0)
Initialize Okt from konlpy (0.6.0)
Initialize khaiii (0.4)
                kiwi   komoran  mecab   kkma   hannanum okt     khaiii
chungcheong.txt 0.8838  0.5989  0.7090  0.5547  0.5096  0.3507  0.7324
gangwon.txt     0.8233  0.5379  0.6263  0.4898  0.4729  0.2917  0.6433
gyeonggi.txt    0.9038  0.6669  0.7489  0.6897  0.6038  0.3870  0.7755
gyeongsang.txt  0.8131  0.5484  0.6366  0.5226  0.5426  0.3187  0.6760
hamgyeong.txt   0.8183  0.3547  0.4627  0.3634  0.3759  0.2028  0.5036
hwanghae.txt    0.8053  0.4763  0.5651  0.4595  0.4225  0.2760  0.6224
jeju.txt        0.7964  0.3982  0.5139  0.3723  0.4258  0.2450  0.5020
jeolla.txt      0.8300  0.5083  0.6373  0.4769  0.5084  0.3040  0.6526
pyeongan.txt    0.8217  0.4159  0.5057  0.3920  0.4261  0.2736  0.5150
<Average>       0.8329  0.5006  0.6006  0.4801  0.4764  0.2944  0.6247
```

참고로 Kiwi의 방언 옵션을 활성화하지 않고 분석한 결과는 다음과 같습니다.

```console
$ python tokenize.py dataset/dialect/*.txt --target=kiwi --error_output_dir=errors/
Initialize kiwipiepy (0.22.0)
                kiwi
chungcheong.txt 0.7531
gangwon.txt     0.6858
gyeonggi.txt    0.8253
gyeongsang.txt  0.6700
hamgyeong.txt   0.5008
hwanghae.txt    0.6413
jeju.txt        0.5218
jeolla.txt      0.6642
pyeongan.txt    0.5089
<Average>       0.6412
```