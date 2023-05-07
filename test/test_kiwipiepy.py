import os
import sys
import re

from kiwipiepy import Kiwi, TypoTransformer, basic_typos, MorphemeSet, sw_tokenizer
from kiwipiepy.utils import Stopwords

curpath = os.path.dirname(os.path.abspath(__file__))

class FileReader:
    def __init__(self, path):
        self.path = path

    def __iter__(self):
        yield from open(self.path, encoding='utf-8')

def test_glue_empty():
    kiwi = Kiwi()
    kiwi.glue([])

def test_repr():
    kiwi = Kiwi()
    print(repr(kiwi))

def test_morpheme_set():
    kiwi = Kiwi()
    ms = MorphemeSet(kiwi, ["먹/VV", "사람", ("고맙", "VA")])
    print(repr(ms))
    assert len(ms) == 3

def test_blocklist():
    kiwi = Kiwi()
    tokens = kiwi.tokenize("고마움을")
    assert tokens[0].form == "고마움"
    
    tokens = kiwi.tokenize("고마움을", blocklist=['고마움'])
    assert tokens[0].form == "고맙"

def test_swtokenizer():
    tokenizer = sw_tokenizer.SwTokenizer('Kiwi/test/written.tokenizer.json')
    print(tokenizer.vocab)
    print(tokenizer.config)
    strs = [
        "",
        "한국어에 특화된 토크나이저입니다.", 
        "감사히 먹겠습니당!",
        "노래진 손톱을 봤던걸요.",
        "제임스웹우주천체망원경",
        "그만해여~",
    ]
    for s in strs:
        token_ids = tokenizer.encode(s)
        token_ids, offset = tokenizer.encode(s, return_offsets=True)
        decoded = tokenizer.decode(token_ids)
        assert s == decoded

def test_swtokenizer_batch():
    tokenizer = sw_tokenizer.SwTokenizer('Kiwi/test/written.tokenizer.json')
    strs = [
        "",
        "한국어에 특화된 토크나이저입니다.", 
        "감사히 먹겠습니당!",
        "노래진 손톱을 봤던걸요.",
        "제임스웹우주천체망원경",
        "그만해여~",
    ]
    for token_ids, s in zip(tokenizer.encode(strs), strs):
        decoded = tokenizer.decode(token_ids)
        assert s == decoded

def test_swtokenizer_morph():
    tokenizer = sw_tokenizer.SwTokenizer('Kiwi/test/written.tokenizer.json')
    
    token_ids = tokenizer.encode("한국어에 특화된 토크나이저입니다.")

    morphs = [
        ('한국어', 'NNP', False), 
        ('에', 'JKB',), 
        ('특화', 'NNG', True), 
        ('되', 'XSV',), 
        ('ᆫ', 'ETM', False), 
        ('토크나이저', 'NNG', True), 
        ('이', 'VCP', False), 
        ('ᆸ니다', 'EF',), 
        ('.', 'SF', False),
    ]

    token_ids_from_morphs = tokenizer.encode_from_morphs(morphs)

    assert (token_ids == token_ids_from_morphs).all()

    token_ids_from_morphs, offsets = tokenizer.encode_from_morphs(morphs, return_offsets=True)

    assert offsets.tolist() == [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [5, 6], [5, 6], [6, 7], [7, 8], [8, 9]]

def test_swtokenizer_tokenize_encode():
    tokenizer = sw_tokenizer.SwTokenizer('Kiwi/test/written.tokenizer.json')
    sents = [
        "한국어에 특화된 토크나이저입니다.",
        "홈페이지는 https://bab2min.github.io/kiwipiepy 입니다."
    ]
    
    for sent in sents:
        ref_token_ids = tokenizer.encode(sent)
        ref_morphs = tokenizer.kiwi.tokenize(sent, normalize_coda=True, z_coda=True)
        morphs, token_ids, offset = tokenizer.tokenize_encode(sent, return_offsets=True)
        assert [m.tagged_form for m in morphs] == [m.tagged_form for m in ref_morphs]
        assert token_ids.tolist() == ref_token_ids.tolist()

    for (morphs, token_ids, offset), sent in zip(tokenizer.tokenize_encode(sents, return_offsets=True), sents):
        ref_token_ids = tokenizer.encode(sent)
        ref_morphs = tokenizer.kiwi.tokenize(sent, normalize_coda=True, z_coda=True)
        assert [m.tagged_form for m in morphs] == [m.tagged_form for m in ref_morphs]
        assert token_ids.tolist() == ref_token_ids.tolist()

def test_swtokenizer_morph_offset():
    tokenizer = sw_tokenizer.SwTokenizer('Kiwi/tokenizers/kor.32k.json')
    morphs = [
        ('칼슘', 'NNG', True), 
        ('·', 'SP', False), 
        ('마그네슘', 'NNG', False), 
        ('등', 'NNB', True), 
        ('이', 'JKS', False), 
        ('많이', 'MAG', True), 
        ('함유', 'NNG', True), 
        ('되', 'XSV', False), 
        ('어', 'EC', False), 
        ('있', 'VX', True), 
        ('어', 'EC', False)
    ]
    token_ids, offsets = tokenizer.encode_from_morphs(morphs, return_offsets=True)
    assert len(token_ids) == len(offsets)
    assert offsets[2:7].tolist() == [[2, 3], [2, 3], [2, 3], [2, 3], [2, 3]]

def test_swtokenizer_trainer_empty():
    config = sw_tokenizer.SwTokenizerConfig()
    sw_tokenizer.SwTokenizer.train(
        'test.json', 
        [], 
        config,
        4000,
    )

def test_swtokenizer_trainer_small():
    config = sw_tokenizer.SwTokenizerConfig()
    sw_tokenizer.SwTokenizer.train(
        'test.json', 
        ["이런 저런 문장", "이렇게 저렇게 처리해서", "이렇고 저렇고", "이런 저런 결과를", "얻었다"], 
        config,
        4000,
    )

def test_swtokenizer_trainer_digits():
    kiwi = Kiwi(num_workers=1)
    config = sw_tokenizer.SwTokenizerConfig()

    tokenizer = sw_tokenizer.SwTokenizer.train(
        'test.json', 
        [f"드디어 제{i}회 평가" for i in range(1, 1000)], 
        config,
        4000,
        kiwi=kiwi,
        prevent_mixed_digit_tokens=False,
    )
    mixed_digit = [k for k in tokenizer.vocab if re.search(r'제[0-9]|[0-9]회', k)]
    assert len(mixed_digit) > 0

    tokenizer = sw_tokenizer.SwTokenizer.train(
        'test.json', 
        [f"드디어 제{i}회 평가" for i in range(1, 1000)], 
        config,
        4000,
        kiwi=kiwi,
        prevent_mixed_digit_tokens=True,
    )
    mixed_digit = [k for k in tokenizer.vocab if re.search(r'제[0-9]|[0-9]회', k)]
    assert len(mixed_digit) == 0

def test_swtokenizer_trainer():
    import itertools

    config = sw_tokenizer.SwTokenizerConfig()
    sw_tokenizer.SwTokenizer.train(
        'test.json', 
        itertools.chain.from_iterable(open(f, encoding='utf-8') for f in (
            'kiwipiepy/documentation.md', 
            'kiwipiepy/_wrap.py', 
        )), 
        config,
        4000,
    )

def test_swtokenizer_trainer_multiple_vocab_sizes():
    import itertools

    config = sw_tokenizer.SwTokenizerConfig()
    sw_tokenizer.SwTokenizer.train(
        ['test.json', 'test2.json', 'test3.json'], 
        itertools.chain.from_iterable(open(f, encoding='utf-8') for f in (
            'kiwipiepy/documentation.md', 
            'kiwipiepy/_wrap.py', 
        )), 
        config,
        [4000, 2000, 1000],
    )

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
    kiwi = Kiwi(load_typo_dict=False)
    ores, oscore = kiwi.analyze("했어요! 하잖아요! 할까요? 좋아요!")[0]

    assert len(kiwi.add_re_rule("EF", r"요$", "용", score=0)) > 0
    res, score = kiwi.analyze("했어용! 하잖아용! 할까용? 좋아용!")[0]
    assert score == oscore

    kiwi = Kiwi(load_typo_dict=False)
    assert len(kiwi.add_re_rule("EF", r"요$", "용", score=-1)) > 0
    res, score = kiwi.analyze("했어용! 하잖아용! 할까용? 좋아용!")[0]
    assert score == oscore - 4

def test_add_pre_analyzed_word():
    kiwi = Kiwi()
    ores = kiwi.tokenize("팅겼어")

    try:
        kiwi.add_pre_analyzed_word("팅겼어", [("팅기", "VV"), "었/EP", "어/EF"])
        raise AssertionError("expected to raise `ValueError`")
    except ValueError:
        pass
    except:
        raise

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

def test_join():
    kiwi = Kiwi()
    tokens = kiwi.tokenize("이렇게 형태소로 분해된 문장을 다시 합칠 수 있을까요?")
    
    assert kiwi.join(tokens) == "이렇게 형태소로 분해된 문장을 다시 합칠 수 있을까요?"

    assert (kiwi.join([("왜", "MAG"), ("저", "NP"), ("한테", "JKB"), ("묻", "VV"), ("어요", "EF")]) 
        == "왜 저한테 물어요"
    )
    assert (kiwi.join([("왜", "MAG"), ("저", "NP"), ("한테", "JKB"), ("묻", "VV-R"), ("어요", "EF")])
        == "왜 저한테 묻어요"
    )
    assert (kiwi.join([("왜", "MAG"), ("저", "NP"), ("한테", "JKB"), ("묻", "VV-I"), ("어요", "EF")])
        == "왜 저한테 물어요"
    )

def test_bug_87():
    text = "한글(韓㐎[1], 영어: Hangeul[2]또는 Hangul[3])은 한국어의 공식문자로서, 세종이 한국어를 표기하기 위하여 창제한 문자인 '훈민정음'(訓民正音)을 20세기 초반 이후 달리 이르는 명칭이다.[4][5] 한글이란 이름은 주시경 선생과 국어연구학회 회원들에 의해 지어진것으로 알려져 있으며[6][7][8][9] 그 뜻은 '으뜸이 되는 큰글', '오직 하나뿐인 큰글', '한국인의 글자'이다.[6][10] 한글의 또 다른 별칭으로는 정음(正音), 언문(諺文)[11], 언서(諺書), 반절(反切), 암클, 아햇글, 가갸글, 국문(國文)[12] 등이 있다.[5]"
    kiwi = Kiwi()
    print(kiwi.join(kiwi.tokenize(text)))

def test_typo_transformer():
    print(basic_typos.generate("안돼"))

def test_typo_correction():
    if sys.maxsize <= 2**32:
        print("[skipped this test in 32bit OS.]", file=sys.stderr)
        return
    kiwi = Kiwi(typos='basic')
    ret = kiwi.tokenize("외않됀대?")
    assert ret[0].form == '왜'
    assert ret[1].form == '안'
    assert ret[2].form == '되'
    assert ret[3].form == 'ᆫ대'
    print(ret)

def test_sbg():
    kiwi = Kiwi(model_type='knlm')
    print(kiwi.tokenize('이 번호로 전화를 이따가 꼭 반드시 걸어.'))
    kiwi = Kiwi(model_type='sbg')
    print(kiwi.tokenize('이 번호로 전화를 이따가 꼭 반드시 걸어.'))

def test_issue_92():
    if sys.maxsize <= 2**32:
        print("[skipped this test in 32bit OS.]", file=sys.stderr)
        return
    kiwi = Kiwi(typos='basic')
    try:
        kiwi.join(kiwi.analyze('쁘'))
        raise AssertionError("expected to raise `ValueError`")
    except ValueError:
        pass
    except:
        raise

    kiwi.join([('사랑','NNG')])

def test_unicode():
    kiwi = Kiwi()
    print(repr(kiwi.tokenize("결 론 189   참고문헌 191   󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝󰠝")))
