import os

from kiwipiepy import Kiwi
from kiwipiepy.utils import Stopwords

curpath = os.path.dirname(os.path.abspath(__file__))

class FileReader:
    def __init__(self, path):
        self.path = path

    def __iter__(self):
        yield from open(self.path, encoding='utf-8')

def test_analyze_single():
    kiwi = Kiwi()
    for line in open(curpath + '/test_corpus/constitution.txt', encoding='utf-8'):
        toks, score = kiwi.analyze(line)[0]
    for t in toks:
        print(t.form, t.tag, t.start, t.end, t.len, t.id, t.base_form, t.base_id)
        break

def test_extract_words():
    kiwi = Kiwi()
    ret = kiwi.extract_words(FileReader(curpath + '/test_corpus/constitution.txt'), min_cnt=2)
    print(ret)

def test_perform():
    kiwi = Kiwi()
    for res in kiwi.perform(FileReader(curpath + '/test_corpus/constitution.txt'), min_cnt=2):
        print(res)
        break

def test_tweet():
    kiwi = Kiwi()
    kiwi.analyze('''#바둑#장기#오목 귀요미#보드판🐥
#어린이임블리의 놀이였는데, 이제는 가물갸물🙄모르겠
장이요~멍이요~ㅎㅎㅎ다시 한 번 재미를 붙여 보까ㅎ
할 일이 태산인데😭, 하고 싶은건 무궁무진🤦‍♀️ 큰 일이다''')

def test_new_analyze_multi():
    kiwi = Kiwi()
    for res in kiwi.analyze(open(curpath + '/test_corpus/constitution.txt', encoding='utf-8')):
        pass

def test_bug_33():
    kiwi = Kiwi()
    kiwi.add_user_word('김갑갑', 'NNP')
    print(kiwi.analyze("김갑갑 김갑갑 김갑갑"))

def test_bug_38():
    text = "이 예쁜 꽃은 독을 품었지만 진짜 아름다움을 가지고 있어요"
    kiwi = Kiwi(integrate_allomorph=True)
    print(kiwi.analyze(text))
    kiwi = Kiwi(integrate_allomorph=False)
    print(kiwi.analyze(text))

def test_property():
    kiwi = Kiwi()
    print(kiwi.integrate_allomorph)
    kiwi.integrate_allomorph = False
    print(kiwi.integrate_allomorph)
    print(kiwi.cutoff_threshold)
    kiwi.cutoff_threshold = 1
    print(kiwi.cutoff_threshold)

def test_stopwords():
    kiwi = Kiwi()
    tokens, _ = kiwi.analyze('불용어 처리 테스트 중입니다 '
                             '우리는 강아지를 좋아한다 쟤도 강아지를 좋아한다 '
                             '지금은 2021년 11월이다.')[0]
    stopwords = Stopwords()
    print(set(tokens) - set(stopwords.filter(tokens)))
    filename = curpath + '/test_corpus/custom_stopwords.txt'
    stopwords = Stopwords(filename)

    stopwords.add(('강아지', 'NNP'))
    assert (('강아지', 'NNP') in stopwords) == True

    stopwords.remove(('강아지', 'NNP'))
    assert (('강아지', 'NNP') in stopwords) == False
    print(set(tokens) - set(stopwords.filter(tokens)))

def test_tokenize():
    kiwi = Kiwi()
    text = "다녀온 후기\n\n<강남 토끼정에 다녀왔습니다.> 음식도 맛있었어요 다만 역시 토끼정 본점 답죠?ㅎㅅㅎ 그 맛이 크으.. 아주 맛있었음...! ^^"
    tokens = kiwi.tokenize(text, normalize_coda=True)
    print(tokens)

    tokens_by_sent = kiwi.tokenize(text, normalize_coda=True, split_sents=True)
    for tokens in tokens_by_sent:
        print(tokens)

def test_tokenize_with_stopwords():
    kiwi = Kiwi()
    stopwords = Stopwords()
    tokens = kiwi.tokenize("[^^ 우리는 강아지를 좋아한다.]", stopwords=stopwords)

    assert tokens[0].form == '강아지'
    assert tokens[1].form == '좋아하'

def test_split_into_sents():
    kiwi = Kiwi()
    text = "다녀온 후기\n\n<강남 토끼정에 다녀왔습니다.> 음식도 맛있었어요 다만 역시 토끼정 본점 답죠?ㅎㅅㅎ 그 맛이 크으.. 아주 맛있었음...! ^^"
    sents = kiwi.split_into_sents(text, normalize_coda=True)
    assert len(sents) == 6

    assert sents[0].text == "다녀온 후기"
    assert sents[1].text == "<강남 토끼정에 다녀왔습니다.>"
    assert sents[2].text == "음식도 맛있었어요"
    assert sents[3].text == "다만 역시 토끼정 본점 답죠?ㅎㅅㅎ"
    assert sents[4].text == "그 맛이 크으.."
    assert sents[5].text == "아주 맛있었음...! ^^"

def test_add_rule():
    kiwi = Kiwi()
    ores, oscore = kiwi.analyze("했어요! 하잖아요! 할까요?")[0]

    assert len(kiwi.add_re_rule("EF", r"요$", "용", score=0)) > 0
    res, score = kiwi.analyze("했어용! 하잖아용! 할까용?")[0]
    assert score == oscore

    kiwi = Kiwi()
    assert len(kiwi.add_re_rule("EF", r"요$", "용", score=-1)) > 0
    res, score = kiwi.analyze("했어용! 하잖아용! 할까용?")[0]
    assert score == oscore - 3

def test_add_pre_analyzed_word():
    kiwi = Kiwi()
    ores = kiwi.tokenize("팅겼어")

    try:
        kiwi.add_pre_analyzed_word("팅겼어", [("팅기", "VV"), "었/EP", "어/EF"])
        raise AssertionError("expected to raise `ValueError`")
    except ValueError:
        pass
    except:
        raise AssertionError("expected to raise `ValueError`")

    kiwi.add_user_word("팅기", "VV", orig_word="튕기")
    kiwi.add_pre_analyzed_word("팅겼어", [("팅기", "VV", 0, 2), ("었", "EP", 1, 2), ("어", "EF", 2, 3)])

    res = kiwi.tokenize("팅겼어...")

    assert res[0].form == "팅기" and res[0].tag == "VV" and res[0].start == 0 and res[0].end == 2
    assert res[1].form == "었" and res[1].tag == "EP" and res[1].start == 1 and res[1].end == 2
    assert res[2].form == "어" and res[2].tag == "EF" and res[2].start == 2 and res[2].end == 3
    assert res[3].form == "..." and res[3].tag == "SF" and res[3].start == 3 and res[3].end == 6

def test_space_tolerance():
    kiwi = Kiwi()
    s = "띄 어 쓰 기 문 제 가 있 습 니 다"
    kiwi.space_tolerance = 0
    print(kiwi.tokenize(s))
    kiwi.space_tolerance = 1
    print(kiwi.tokenize(s))
    kiwi.space_tolerance = 2
    print(kiwi.tokenize(s))
    kiwi.space_tolerance = 3
    print(kiwi.tokenize(s))

def test_space():
    kiwi = Kiwi()
    res0 = kiwi.space("띄어쓰기없이작성된텍스트네이걸교정해줘.")
    assert res0 == "띄어쓰기 없이 작성된 텍스트네 이걸 교정해 줘."

    res1 = kiwi.space(" 띄어쓰기없이 작성된텍스트(http://github.com/bab2min/kiwipiepy )네,이걸교정해줘. ")
    assert res1 == " 띄어쓰기 없이 작성된 텍스트(http://github.com/bab2min/kiwipiepy )네, 이걸 교정해 줘. "

    res2 = kiwi.space("<Kiwipiepy>는 형태소 분석기이에요~ 0.11.0 버전이 나왔어요.")
    assert res2 == "<Kiwipiepy>는 형태소 분석기이에요~ 0.11.0 버전이 나왔어요."

    res3 = kiwi.space("<Kiwipiepy>는 형 태 소 분 석 기 이 에 요~ 0.11.0 버 전 이 나 왔 어 요 .", reset_whitespace=True)
    assert res3 == "<Kiwipiepy>는 형태소 분석기이에요~ 0.11.0 버전이 나왔어요."

    res_a = list(kiwi.space([
        "띄어쓰기없이작성된텍스트네이걸교정해줘.",
        " 띄어쓰기없이 작성된텍스트(http://github.com/bab2min/kiwipiepy )네,이걸교정해줘. ",
        "<Kiwipiepy>는 형태소 분석기이에요~ 0.11.0 버전이 나왔어요.",
    ]))
    assert res_a == [res0, res1, res2]


def test_glue():
    chunks = """KorQuAD 2.0은 총 100,000+ 쌍으로 구성된 한국어 질의응답 데이터셋이다. 기존 질의응답 표준 데이
터인 KorQuAD 1.0과의 차이점은 크게 세가지가 있는데 첫 번째는 주어지는 지문이 한두 문단이 아닌 위
키백과 한 페이지 전체라는 점이다. 두 번째로 지문에 표와 리스트도 포함되어 있기 때문에 HTML tag로
구조화된 문서에 대한 이해가 필요하다. 마지막으로 답변이 단어 혹은 구의 단위뿐 아니라 문단, 표, 리
스트 전체를 포괄하는 긴 영역이 될 수 있다. Baseline 모델로 구글이 오픈소스로 공개한 BERT
Multilingual을 활용하여 실험한 결과 F1 스코어 46.0%의 성능을 확인하였다. 이는 사람의 F1 점수
85.7%에 비해 매우 낮은 점수로, 본 데이터가 도전적인 과제임을 알 수 있다. 본 데이터의 공개를 통해
평문에 국한되어 있던 질의응답의 대상을 다양한 길이와 형식을 가진 real world task로 확장하고자 한다""".split('\n')
    
    kiwi = Kiwi()
    ret, space_insertions = kiwi.glue(chunks, return_space_insertions=True)
    assert space_insertions == [False, False, True, False, True, True, True]
