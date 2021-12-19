import os
curpath = os.path.dirname(os.path.abspath(__file__))

from kiwipiepy import Kiwi
from kiwipiepy.utils import Stopwords

def test_analyze_single():
    kiwi = Kiwi()
    for line in open(curpath + '/test_corpus/constitution.txt', encoding='utf-8'):
        toks, score = kiwi.analyze(line)[0]
    for t in toks:
        print(t.form, t.tag, t.start, t.end, t.len)
        break


class FileReader:
    def __init__(self, path):
        self.path = path

    def __iter__(self):
        yield from open(self.path, encoding='utf-8')

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
