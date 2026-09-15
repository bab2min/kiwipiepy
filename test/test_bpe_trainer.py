import json

import pytest

from kiwipiepy import Kiwi
from kiwipiepy.bpe_trainer import train_bpe_tokenizer


SAMPLE_TEXTS = [
    "한국어 형태소 분석기 키위입니다.",
    "키위는 한국어 텍스트를 형태소 단위로 분석합니다.",
    "형태소 분석 결과를 이용해 토크나이저를 학습합니다.",
    "BPE 토크나이저는 자주 등장하는 문자열을 하나의 토큰으로 병합합니다.",
    "숫자 12345 와 반복 문자 ㅋㅋㅋㅋㅋ 도 포함되어 있습니다.",
]

VOCAB_SIZE = 500


def _corpus(repeat=40):
    for _ in range(repeat):
        yield from SAMPLE_TEXTS


def _hf_token(text):
    '''토크나이저 JSON의 어휘에 적히는 GPT-2 방식의 바이트->유니코드 표기로 바꾼다.'''
    printable = list(range(ord('!'), ord('~') + 1)) + list(range(ord('¡'), ord('¬') + 1)) + list(range(ord('®'), ord('ÿ') + 1))
    table = {b: chr(b) for b in printable}
    n = 0
    for b in range(256):
        if b not in table:
            table[b] = chr(256 + n)
            n += 1
    return ''.join(table[b] for b in text.encode('utf-8'))


def _train(tmp_path, name='tokenizer.json', **kwargs):
    save_path = str(tmp_path / name)
    kwargs.setdefault('vocab_size', VOCAB_SIZE)
    kwargs.setdefault('show_progress', False)
    train_bpe_tokenizer(save_path, _corpus(), **kwargs)
    with open(save_path, encoding='utf-8') as f:
        return save_path, json.load(f)


def test_train_bpe_tokenizer(tmp_path):
    _, obj = _train(tmp_path)

    assert obj['model']['type'] == 'BPE'
    vocab = obj['model']['vocab']

    assert 256 <= len(vocab) <= VOCAB_SIZE
    assert len(set(vocab.values())) == len(vocab), "Duplicated Token IDs"
    assert obj['model']['merges'], "No merge rules found"
    assert obj['pre_tokenizer']['type'] == 'ByteLevel'
    
    assert obj['normalizer'] is None


def test_add_prefix_space_is_saved(tmp_path):
    _, off = _train(tmp_path, 'off.json', add_prefix_space=False)
    _, on = _train(tmp_path, 'on.json', add_prefix_space=True)

    assert off['pre_tokenizer']['add_prefix_space'] is False
    assert on['pre_tokenizer']['add_prefix_space'] is True


@pytest.mark.parametrize('jamo_alphabet', ['none', 'modern_only', 'modern_only_with_nfd', 'all', 'all_with_nfd'])
def test_jamo_alphabet_records_normalizer(tmp_path, jamo_alphabet):
    # 'all'은 자모와 그 접두 토큰만으로 vocab_size 최소 631 이상을 요구
    vocab_size = 1000
    _, obj = _train(tmp_path, f'{jamo_alphabet}.json', jamo_alphabet=jamo_alphabet, vocab_size=vocab_size)

    assert obj['model']['type'] == 'BPE'
    assert len(obj['model']['vocab']) <= vocab_size
    if jamo_alphabet.endswith('_with_nfd'):
        assert obj['normalizer'] == {'type': 'NFD'}
    elif jamo_alphabet == 'none':
        assert obj['normalizer'] is None
    else:
        assert obj['normalizer'] == {'type': 'nfd_for_hangul'}


@pytest.mark.parametrize('jamo_alphabet', ['modern_only', 'all'])
def test_all_jamo_alphabet_pins_archaic_jamo(tmp_path, jamo_alphabet):
    _, obj = _train(tmp_path, f'{jamo_alphabet}.json', jamo_alphabet=jamo_alphabet, vocab_size=1000)
    vocab = obj['model']['vocab']

    assert _hf_token(chr(0x1112)) in vocab
    archaic = [chr(c) for c in (0x1113, 0x119E, 0xA960, 0xD7B0)]
    if jamo_alphabet == 'all':
        blocks = (range(0x1100, 0x1200), range(0xA960, 0xA980), range(0xD7B0, 0xD800))
        assert all(_hf_token(chr(c)) in vocab for block in blocks for c in block)
    else:
        assert not any(_hf_token(c) in vocab for c in archaic)

def test_normalizer_patch_preserves_content(tmp_path):
    _, plain = _train(tmp_path, 'plain.json')
    _, jamo = _train(tmp_path, 'jamo.json', jamo_alphabet='modern_only')

    assert plain.keys() == jamo.keys()
    for key in ('added_tokens', 'decoder', 'pre_tokenizer', 'version'):
        assert plain[key] == jamo[key]


def test_error_in_texts_propagates(tmp_path):
    class Boom(Exception):
        pass

    def broken_texts():
        yield from _corpus(repeat=5)
        raise Boom('Failed to read texts')

    with pytest.raises(Boom):
        train_bpe_tokenizer(
            str(tmp_path / 'broken.json'),
            broken_texts(),
            vocab_size=VOCAB_SIZE,
            show_progress=False,
        )


def test_pretokenize_requires_kiwi(tmp_path):
    for option in ('pretokenize_j', 'pretokenize_e', 'pretokenize_vcp', 'pretokenize_xsv'):
        with pytest.raises(ValueError):
            _train(tmp_path, **{option: True})


def test_invalid_jamo_alphabet(tmp_path):
    with pytest.raises(ValueError):
        _train(tmp_path, jamo_alphabet='nfc')


def test_pretokenize_with_kiwi(tmp_path):
    kiwi = Kiwi()
    _, obj = _train(tmp_path, kiwi=kiwi, pretokenize_j=True, pretokenize_e=True)

    assert obj['model']['type'] == 'BPE'
    assert obj['model']['merges']


def test_output_is_loadable_by_huggingface(tmp_path):
    tokenizers = pytest.importorskip('tokenizers')

    save_path, _ = _train(tmp_path)
    tokenizer = tokenizers.Tokenizer.from_file(save_path)

    encoded = tokenizer.encode(SAMPLE_TEXTS[0])
    assert encoded.ids
    assert tokenizer.decode(encoded.ids) == SAMPLE_TEXTS[0]
