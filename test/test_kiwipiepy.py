import os
import sys
import re
import tempfile
import itertools

from kiwipiepy import Kiwi, TypoTransformer, basic_typos, MorphemeSet, sw_tokenizer, PretokenizedToken, extract_substrings
from kiwipiepy.utils import Stopwords

curpath = os.path.dirname(os.path.abspath(__file__))

class FileReader:
    def __init__(self, path):
        self.path = path

    def __iter__(self):
        yield from open(self.path, encoding='utf-8')

def test_extract_substrings():
    s = ("자, 너 오늘 하루 뭐 했니? "
		"난 오늘 하루가 좀 단순했지. 음 뭐 했는데? "
		"아침에 수업 받다가 오구 그~ 학생이, 응. 미찌 학생인데, "
		"음. 되게 귀엽게 생겼다, 음. 되게 웃으면 곰돌이 인형같이 생겼어. "
		"곰돌이? 응 아니 곰돌이도 아니구, 어쨌든 무슨 인형같이 생겼어, "
		"펜더곰 그런 거, 왜 이렇게 닮은 사람을 대 봐. "
		"내가 아는 사람 중에서, 한무 한무? 금붕어잖아? "
		"맞어 눈도 이렇게 톡 튀어나오구, 어. 한무? 조금 잘 생긴 한무. "
		"잘 생긴 게 아니라 귀여운 한무. 귀여운 한무? "
		"어 학원에서 별명도 귀여운 한무였어. "
		"응. 눈이 똥그래 가지고 그래? 어. 좀 특이한 사람이구나.")
    substrings = extract_substrings(s, min_cnt=2, min_length=2, longest_only=True, stop_chr=' ')
    print(substrings)
    assert len(substrings) == 23

def test_false_positive_sb():
    kiwi = Kiwi()
    for s in [
        "하다. 가운데 비닐을 요렇게 벗겨주고요!",
		"자, 이것이 `열쇠`다.`` 암상인 앞의 캡슐이 열리며 그곳에서 새로운 파워업 아이템 `쿠나이`를 얻을 수 있다.",
		"기계는 명령만 듣는다.라는 생각이 이제 사람들에게 완전히 정착이 되었다는 상황인데, 그럴싸하죠",
		"후반 빨간 모아이들의 공격은 엄청나게 거세다.상하로 점프하며 이온링을 발사하는 중보스 모아이상들.",
		"또 전화세, 전기세, 보험료등 월 정기지출도 지출통장으로 바꾼 다. 셋째, 물건을 살땐 무조건 카드로 긁는다.",
		"에티하드항공이 최고의 이코노미 클래스 상을 두 번째로 받은 해는 2020년이다. 이전에는 2012년과 2013년에 최고의 이코노미 클래스 상을 수상한 적이 있어요.",
		"은미희는 새로운 삶을 시작하기로 결심한 후 6년간의 습작기간을 거쳐 1996년 단편 《누에는 고치 속에서 무슨 꿈을 꾸는가》로 전남일보 신춘문예 당선됐다. 따라서 은미희가 《누에는 고치 속에서 무슨 꿈을 꾸는가》로 상을 받기 전에 연습 삼아 소설을 집필했던 기간은 6년이다.",
		"도서전에서 관람객의 관심을 받을 것으로 예상되는 프로그램으로는 '인문학 아카데미'가 있어요. 이 프로그램에서는 유시민 전 의원, 광고인 박웅현 씨 등이 문화 역사 미학 등 다양한 분야에 대해 강의할 예정이다. 또한, '북 멘토 프로그램'도 이어져요. 이 프로그램에서는 각 분야 전문가들이 경험과 노하우를 전수해 주는 프로그램으로, 시 창작(이정록 시인), 번역(강주헌 번역가), 북 디자인(오진경 북디자이너) 등의 분야에서 멘토링이 이뤄져요.",
    ]:
        assert all(t.tag != 'SB' for t in kiwi.tokenize(s))

def test_repr():
    kiwi = Kiwi()
    print(repr(kiwi))

def test_morpheme_set():
    kiwi = Kiwi()
    ms = MorphemeSet(kiwi, ["먹/VV", "사람", ("고맙", "VA")])
    print(repr(ms))
    assert len(ms) == 3

def test_load_user_dictionary():
    kiwi = Kiwi()
    try:
        raised = False
        kiwi.load_user_dictionary('non-existing-file.txt')
    except OSError as e:
        raised = True
        print(e)
    finally:
        assert raised

    with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', delete=False) as f:
        f.write('잘못된 포맷의 파일입니다\n')
    try:
        raised = False
        kiwi.load_user_dictionary(f.name)
    except ValueError as e:
        raised = True
        print(e)
    finally:
        assert raised

def test_issue_158():
    print(len(Kiwi().tokenize('보통' * 40000)))

def test_list_all_scripts():
    kiwi = Kiwi()
    print(kiwi.list_all_scripts())

def test_blocklist():
    kiwi = Kiwi()
    tokens = kiwi.tokenize("고마움을")
    assert tokens[0].form == "고마움"
    
    tokens = kiwi.tokenize("고마움을", blocklist=['고마움'])
    assert tokens[0].form == "고맙"

def test_pretokenized():
    kiwi = Kiwi(load_multi_dict=False)
    text = "드디어패트와 매트가 2017년에 국내 개봉했다. 패트와매트는 2016년..."

    res = kiwi.tokenize(text, pretokenized=[
        (3, 9),
        (11, 16),
        (34, 39)
    ])
    assert res[1].form == "패트와 매트"
    assert res[1].tag == "NNP"
    assert res[3].form == "2017년"
    assert res[3].tag == "NNP"
    assert res[13].form == "2016년"
    assert res[13].tag == "NNP"

    res = kiwi.tokenize(text, pretokenized=[
        (3, 9),
        (11, 16, 'NNG'),
        (34, 39, 'NNG')
    ])
    assert res[3].form == "2017년"
    assert res[3].tag == "NNG"
    assert res[13].form == "2016년"
    assert res[13].tag == "NNG"

    res = kiwi.tokenize(text, pretokenized=[
        (27, 29, PretokenizedToken('페트', 'NNB', 0, 2)),
        (30, 32),
        (21, 24, [PretokenizedToken('개봉하', 'VV', 0, 3), PretokenizedToken('었', 'EP', 2, 3)])
    ])
    assert res[7].form == "개봉하"
    assert res[7].tag == 'VV'
    assert res[7].start == 21
    assert res[7].len == 3
    assert res[8].form == "었"
    assert res[8].tag == 'EP'
    assert res[8].start == 23
    assert res[8].len == 1
    assert res[11].form == "페트"
    assert res[11].tag == 'NNB'
    assert res[13].form == "매트"
    assert res[13].tag == 'NNG'

    res = kiwi.tokenize(text, pretokenized=lambda x:[i.span() for i in re.finditer(r'패트와 ?매트', x)])
    assert res[1].form == "패트와 매트"
    assert res[1].span == (3, 9)
    assert res[12].form == "패트와매트"
    assert res[12].span == (27, 32)

    res = kiwi.tokenize(text, pretokenized=[
        (3, 5, PretokenizedToken('페트', 'NNB', 0, 2)),
    ])
    assert res[1].form == '페트'

    try:
        is_raised = False
        kiwi.tokenize("some text", pretokenized=[(-1, 0)])
    except ValueError as e:
        is_raised = True
        print(e)
    finally:
        assert is_raised

    try:
        is_raised = False
        kiwi.tokenize("some text", pretokenized=[(0, 1000)])
    except ValueError as e:
        is_raised = True
        print(e)
    finally:
        assert is_raised

    try:
        is_raised = False
        kiwi.tokenize("some text", pretokenized=[(1, 0)])
    except ValueError as e:
        is_raised = True
        print(e)
    finally:
        assert is_raised

def test_re_word():
    text = '{평만경(平滿景)}이 사람을 시켜 {침향(沈香)} 10냥쭝을 바쳤으므로'

    kiwi = Kiwi()
    res = kiwi.tokenize(text)

    kiwi.add_re_word(r'\{[^}]+\}', 'NNP')

    res = kiwi.tokenize(text)
    assert res[0].form == '{평만경(平滿景)}'
    assert res[0].tag == 'NNP'
    assert res[0].span == (0, 10)
    assert res[1].tag == 'JKS'
    assert res[6].form == '{침향(沈香)}'
    assert res[6].tag == 'NNP'
    assert res[6].span == (19, 27)

    kiwi.clear_re_words()
    kiwi.add_re_word(r'(?<=\{)([^}]+)(?=\})', lambda m:PretokenizedToken(m.group(1), 'NNP', m.span(1)[0] - m.span(0)[0], m.span(1)[1] - m.span(0)[0]))

    res = kiwi.tokenize(text)
    assert res[1].form == '평만경(平滿景)'
    assert res[1].tag == 'NNP'
    assert res[1].span == (1, 9)
    assert res[9].form == '침향(沈香)'
    assert res[9].tag == 'NNP'
    assert res[9].span == (20, 26)

    kiwi.clear_re_words()
    kiwi.add_re_word(r'\{([^}]+)\}', lambda m:PretokenizedToken(m.group(1), 'NNP', m.span(1)[0] - m.span(0)[0], m.span(1)[1] - m.span(0)[0]))

    res = kiwi.tokenize(text)
    assert res[0].form == '평만경(平滿景)'
    assert res[0].tag == 'NNP'
    assert res[0].span == (0, 10)
    assert res[6].form == '침향(沈香)'
    assert res[6].tag == 'NNP'
    assert res[6].span == (19, 27)

    res = kiwi.tokenize(text, pretokenized=[(28, 32)])
    assert res[7].form == '10냥쭝'
    assert res[7].tag == 'NNP'
    assert res[7].span == (28, 32)

    res = kiwi.tokenize(text, pretokenized=[(1, 4)])

def test_user_value():
    kiwi = Kiwi()
    kiwi.add_user_word('사용자단어', user_value=['user_value'])
    kiwi.add_user_word('태그', user_value={'tag':'USER_TAG'})
    kiwi.add_re_rule('NNG', '바보', '밥오', user_value='babo')
    kiwi.add_re_word(r'\{[^}]+\}', 'USER0', user_value={'tag':'SPECIAL'})

    tokens = kiwi.tokenize('사용자단어입니다.')
    assert tokens[0].form == '사용자단어'
    assert tokens[0].user_value == ['user_value']
    assert tokens[1].user_value == None
    assert tokens[2].user_value == None

    tokens = kiwi.tokenize('사용자 태그이다!')
    assert tokens[0].tag == 'NNG'
    assert tokens[1].form == '태그'
    assert tokens[1].tag == 'USER_TAG'
    assert tokens[1].form_tag == ('태그', 'USER_TAG')
    assert tokens[1].tagged_form == '태그/USER_TAG'

    tokens = kiwi.tokenize('밥오..')
    assert tokens[0].form == '밥오'
    assert tokens[0].tag == 'NNG'
    assert tokens[0].user_value == 'babo'

    tokens = kiwi.tokenize('이렇게 {이것}은 특별하다')
    assert tokens[1].form == '{이것}'
    assert tokens[1].tag == 'SPECIAL'
    assert tokens[1].user_value == {'tag':'SPECIAL'}
    assert sum(1 for t in tokens if t.user_value is not None) == 1

    tokens = next(kiwi.tokenize(['{이것}은 특별하다']))
    assert tokens[0].form == '{이것}'
    assert tokens[0].tag == 'SPECIAL'
    assert tokens[0].user_value == {'tag':'SPECIAL'}
    assert sum(1 for t in tokens if t.user_value is not None) == 1

def test_user_value_issue168():
    kiwi = Kiwi()
    text = """마크다운 코드가 섞인 문자열
```python
import kiwipiepy
```
입니다."""

    pat1 = re.compile(r'^```python\n.*?^```', flags=re.DOTALL | re.MULTILINE)
    pat2 = re.compile(r'입니다')

    kiwi.add_re_word(pat1, 'USER1', {'tag':'CODE1'})
    kiwi.add_re_word(pat2, 'USER2', {'tag':'CODE2'})
    tokens = kiwi.tokenize(text)
    assert tokens[-3].tag == 'CODE1'
    assert tokens[-2].tag == 'CODE2'

def test_words_with_space():
    kiwi = Kiwi()
    
    assert kiwi.add_user_word('대학생 선교회', 'NNP')
    res1 = kiwi.tokenize('대학생 선교회')
    res2 = kiwi.tokenize('대학생선교회')
    res3 = kiwi.tokenize('대학생 \t 선교회')
    res4 = kiwi.tokenize('대 학생선교회')
    res5 = kiwi.tokenize('대 학생 선교회')
    res6 = kiwi.tokenize('대학 생선 교회')
    assert len(res1) == 1
    assert len(res2) == 1
    assert len(res3) == 1
    assert len(res4) != 1
    assert len(res5) != 1
    assert len(res6) != 1
    assert res1[0].form == '대학생 선교회'
    assert res2[0].form == '대학생 선교회'
    assert res3[0].form == '대학생 선교회'
    assert res4[0].form != '대학생 선교회'
    assert res5[0].form != '대학생 선교회'
    assert res6[0].form != '대학생 선교회'

    kiwi.space_tolerance = 1
    res1 = kiwi.tokenize('대학생 선교회')
    res2 = kiwi.tokenize('대학생선교회')
    res3 = kiwi.tokenize('대학생 \t 선교회')
    res4 = kiwi.tokenize('대 학생선교회')
    res5 = kiwi.tokenize('대 학생 선교회')
    res6 = kiwi.tokenize('대학 생선 교회')
    assert len(res1) == 1
    assert len(res2) == 1
    assert len(res3) == 1
    assert len(res4) == 1
    assert len(res5) == 1
    assert len(res6) != 1

    kiwi.space_tolerance = 0
    assert kiwi.add_user_word('농협 용인 육가공 공장', 'NNP')
    res1 = kiwi.tokenize('농협 용인 육가공 공장')
    res2 = kiwi.tokenize('농협용인 육가공 공장')
    res3 = kiwi.tokenize('농협 용인육가공 공장')
    res4 = kiwi.tokenize('농협 용인 육가공공장')
    res5 = kiwi.tokenize('농협용인육가공공장')
    res6 = kiwi.tokenize('농협용 인육 가공 공장')
    assert res1[0].form == '농협 용인 육가공 공장'
    assert res2[0].form == '농협 용인 육가공 공장'
    assert res3[0].form == '농협 용인 육가공 공장'
    assert res4[0].form == '농협 용인 육가공 공장'
    assert res5[0].form == '농협 용인 육가공 공장'
    assert res6[0].form != '농협 용인 육가공 공장'
    
    kiwi.space_tolerance = 1
    res2 = kiwi.tokenize('농협용인육 가공 공장')
    res3 = kiwi.tokenize('농협용 인육 가공 공장')
    res4 = kiwi.tokenize('농협용 인육 가공공장')
    assert res2[0].form == '농협 용인 육가공 공장'
    assert res3[0].form != '농협 용인 육가공 공장'
    assert res4[0].form != '농협 용인 육가공 공장'

    kiwi.space_tolerance = 2
    res3 = kiwi.tokenize('농협용 인육 가공 공장')
    res4 = kiwi.tokenize('농협용 인육 가공공장')
    assert res3[0].form == '농협 용인 육가공 공장'
    assert res4[0].form == '농협 용인 육가공 공장'

    res5 = kiwi.tokenize('농협용\n인육 가공\n공장에서')
    assert res5[0].form == '농협 용인 육가공 공장'
    assert res5[0].line_number == 0
    assert res5[1].line_number == 2

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

def test_swtokenizer_offset():
    tokenizer = sw_tokenizer.SwTokenizer('Kiwi/tokenizers/kor.32k.json')
    for sent in [
        '🔴🟠🟡🟢🔵🟣🟤⚫⚪\n'
    ]:
        token_ids, offsets = tokenizer.encode(sent, return_offsets=True)
        assert len(token_ids) == len(offsets)

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
    kiwi = Kiwi(num_workers=0)
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
    assert abs(score - (oscore - 4)) < 1e-3

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

    kiwi.add_pre_analyzed_word('격자판', [('격자', 'NNG'), ('판','NNG')], 100)

    res = kiwi.tokenize("바둑판 모양의 격자판을 펴")
    assert res[3].form == "격자"
    assert res[3].span == (8, 10)
    assert res[4].form == "판"
    assert res[4].span == (10, 11)

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

def test_glue_empty():
    kiwi = Kiwi()
    kiwi.glue([])

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

    assert (kiwi.join([("왜", "MAG"), ("저", "NP"), ("한테", "JKB", True), ("묻", "VV-I"), ("어요", "EF")])
        == "왜 저 한테 물어요"
    )

    assert (kiwi.join([("왜", "MAG"), ("저", "NP"), ("한테", "JKB"), ("묻", "VV-I", False), ("어요", "EF")])
        == "왜 저한테물어요"
    )

def test_join_with_positions():
    kiwi = Kiwi()
    joined, positions = kiwi.join([('🐥', 'SW'), ('하', 'VV'), ('었', 'EP'), ('는데', 'EF')], return_positions=True)
    assert joined == '🐥했는데'
    assert positions == [(0, 1), (1, 2), (1, 2), (2, 4)]

def test_join_edge_cases():
    kiwi = Kiwi()
    for c in [
        '가격이 싼 것이 이것뿐이에요.'
    ]:
        tokens = kiwi.tokenize(c)
        restored = kiwi.join(tokens)
        raw = kiwi.join([(t.form, t.tag) for t in tokens])
        assert c == restored
        assert c == raw

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

def test_continual_typo():
    kiwi = Kiwi(typos='continual')
    tokens = kiwi.tokenize('오늘사무시레서')
    assert tokens[1].form == '사무실'
    assert tokens[2].form == '에서'

    tokens = kiwi.tokenize('지가캤어요')
    assert tokens[0].form == '지각'
    assert tokens[1].form == '하'

    kiwi = Kiwi(typos='basic_with_continual')
    tokens = kiwi.tokenize('웨 지가캤니?')
    assert tokens[0].form == '왜'
    assert tokens[1].form == '지각'
    assert tokens[2].form == '하'

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

def test_template():
    kiwi = Kiwi()
    tpl = kiwi.template("{}가 {}을 {}었다.")

    res = tpl.format(("나", "NP"), ("공부", "NNG"), ("하", "VV"))
    assert res == "내가 공부를 했다."

    res = tpl.format(("너", "NP"), ("밥", "NNG"), ("먹", "VV"))
    assert res == "네가 밥을 먹었다."

    res = tpl.format(("우리", "NP"), ("길", "NNG"), ("묻", "VV-I"))
    assert res == "우리가 길을 물었다."

    res = tpl.format(5, "str", {"dict":"dict"})
    assert res == "5가 str를 {'dict': 'dict'}었다."

    tpl = kiwi.template("{:.5f}가 {!r}을 {}었다.")
    
    res = tpl.format(5, "str", {"dict":"dict"})
    assert res == "5.00000가 'str'를 {'dict': 'dict'}었다."

    caught_error = None
    try:
        res = tpl.format(("우리", "NP"), ("길", "NNG"), ("묻", "VV-I"))
    except ValueError:
        caught_error = True
    assert caught_error

    res = tpl.format(5, ("길", "NNG"), ("묻", "VV-I"))
    assert res == "5.00000가 ('길', 'NNG')를 물었다."

    tpl = kiwi.template("{}가 {}를 {}\ㄴ다.")
    res = tpl.format([("우리", "NP"), ("들", "XSN")], ("길", "NNG"), ("묻", "VV-I"))
    assert res == "우리들이 길을 묻는다."

def test_issue_145():
    kiwi = Kiwi()
    stopwords = Stopwords()
    kiwi.add_user_word('팔이', 'XSV', 10)
    text = "루쉰(노신)의 「아Q정전」은 주인공을 통해 중국민족의 병폐,노예근성을 기탄없이 지적한 작품이다. 날품팔이를 하며 그럭저럭 살아가는 떠돌이 농민 아Q는 자기도 모르는 사이에 혁명의 와중에 휘말려 반란죄로 체포되고 사형선고를 받는다.까닭도 모르고 사형집행 서류에 서명을 하게 되지만 글자를 쓸줄 모르는 일자무식 아Q는 온 힘을 기울여 동그라미를 겨우 그린 후.."
    tokens = kiwi.tokenize(text, split_sents= True, stopwords = stopwords, blocklist = ['껌팔이/NNG','품팔이/NNG', '날품팔이/NNG'])
    assert tokens
