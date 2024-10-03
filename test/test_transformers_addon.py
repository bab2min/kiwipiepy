import tempfile

from transformers import AutoTokenizer
import kiwipiepy.transformers_addon

def test_init():
    global tokenizer
    tokenizer = AutoTokenizer.from_pretrained('test/sample_tokenizer')

def test_basic_tokenize():
    e = tokenizer("가자")
    assert (e['input_ids'] == [2, 75, 130, 3])
    assert (e['attention_mask'] == [1, 1, 1, 1])
    assert (e['token_type_ids'] == [0, 0, 0, 0])

    e = tokenizer("가자", "맞습니다요")
    assert (e['input_ids'] == [2, 75, 130, 3, 282, 64, 157, 3])
    assert (e['attention_mask'] == [1, 1, 1, 1, 1, 1, 1, 1])
    assert (e['token_type_ids'] == [0, 0, 0, 0, 1, 1, 1, 1])

    e = tokenizer(["가자"])
    assert (e['input_ids'] == [[2, 75, 130, 3]])
    assert (e['attention_mask'] == [[1, 1, 1, 1]])
    assert (e['token_type_ids'] == [[0, 0, 0, 0]])
    
    e = tokenizer([("가자", "맞습니다요")])
    assert (e['input_ids'] == [[2, 75, 130, 3, 282, 64, 157, 3]])
    assert (e['attention_mask'] == [[1, 1, 1, 1, 1, 1, 1, 1]])
    assert (e['token_type_ids'] == [[0, 0, 0, 0, 1, 1, 1, 1]])

def test_without_special_tokens():
    e = tokenizer("가자", add_special_tokens=False)
    assert (e['input_ids'] == [75, 130])
    assert (e['attention_mask'] == [1, 1])
    assert (e['token_type_ids'] == [0, 0])

    e = tokenizer("가자", "맞습니다요", add_special_tokens=False)
    assert (e['input_ids'] == [75, 130, 282, 64, 157])
    assert (e['attention_mask'] == [1, 1, 1, 1, 1])
    assert (e['token_type_ids'] == [0, 0, 1, 1, 1])

def test_truncation():
    e = tokenizer("가자가자가자~", "맞습니다요맞구요맞았어요!", truncation=True, max_length=14)
    assert (e['input_ids'] == [2, 75, 130, 75, 130, 75, 3, 282, 64, 157, 282, 1219, 282, 3])
    assert (e['attention_mask'] == [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
    assert (e['token_type_ids'] == [0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1])

    e = tokenizer("가자가자가자~", "맞습니다요맞구요맞았어요!", truncation='only_first', max_length=14)
    assert (e['input_ids'] == [2, 75, 130, 3, 282, 64, 157, 282, 1219, 282, 11, 78, 85, 3])
    assert (e['attention_mask'] == [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
    assert (e['token_type_ids'] == [0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])

    e = tokenizer("가자가자가자~", "맞습니다요맞구요맞았어요!", truncation='only_second', max_length=14)
    assert (e['input_ids'] == [2, 75, 130, 75, 130, 75, 130, 60, 3, 282, 64, 157, 282, 3])
    assert (e['attention_mask'] == [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
    assert (e['token_type_ids'] == [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1])

def test_pad():
    e = tokenizer("가자", padding='max_length', max_length=8)
    assert (e['input_ids'] == [2, 75, 130, 3, 0, 0, 0, 0])
    assert (e['attention_mask'] == [1, 1, 1, 1, 0, 0, 0, 0])
    assert (e['token_type_ids'] == [0, 0, 0, 0, 0, 0, 0, 0])

    e = tokenizer(["가자", "맞습니다요"], padding=True)
    assert (e['input_ids'] == [[2, 75, 130, 3, 0], [2, 282, 64, 157, 3]])
    assert (e['attention_mask'] == [[1, 1, 1, 1, 0], [1, 1, 1, 1, 1]])
    assert (e['token_type_ids'] == [[0, 0, 0, 0, 0], [0, 0, 0, 0, 0]])

def test_offset_mapping():
    e = tokenizer("맞습니다요!", padding='max_length', max_length=8, return_offsets_mapping=True)
    assert (e['input_ids'] == [2, 282, 64, 157, 85, 3, 0, 0])
    assert (e['offset_mapping'] == [(0, 0), (0, 1), (1, 4), (4, 5), (5, 6), (0, 0), (0, 0), (0, 0)])

    e = tokenizer("가자", "맞습니다요", return_offsets_mapping=True)
    assert (e['input_ids'] == [2, 75, 130, 3, 282, 64, 157, 3])
    assert (e['offset_mapping'] == [(0, 0), (0, 1), (1, 2), 
                                    (0, 0), (0, 1), (1, 4), (4, 5), (0, 0)])

    e = tokenizer("가자가자가자~", "맞습니다요맞구요맞았어요!", truncation=True, max_length=14, return_offsets_mapping=True)
    assert (e['input_ids'] == [2, 75, 130, 75, 130, 75, 3, 282, 64, 157, 282, 1219, 282, 3])
    assert (e['offset_mapping'] == [(0, 0), (0, 1), (1, 2), (2, 3), (3, 4), (4, 5), 
                                    (0, 0), (0, 1), (1, 4), (4, 5), (5, 6), (6, 8), (8, 9), (0, 0)])

def test_decode():
    d = tokenizer.decode([2, 75, 130, 3])
    assert d == "[CLS] 가자 [SEP]"

    d = tokenizer.decode([2, 75, 130, 3], skip_special_tokens=True)
    assert d == "가자"

def test_tokenize():
    t = tokenizer.tokenize("맞습니다요!")
    assert t == ["맞/V", "습니다/E", "요/J", "!"]

def test_save_pretrained():
    path = tempfile.gettempdir() + '/test_tokenizer'
    tokenizer.save_pretrained(path)
    new_tokenizer = AutoTokenizer.from_pretrained(path)
    assert new_tokenizer.get_vocab() == tokenizer.get_vocab()

if __name__ == '__main__':
    for k, v in locals().copy().items():
        if k.startswith('test'): v()
