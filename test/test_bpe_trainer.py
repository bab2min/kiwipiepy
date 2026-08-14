import json
import unicodedata

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


@pytest.mark.parametrize('use_jamo_alphabet', [True, 'nfd'])
def test_use_jamo_alphabet_records_normalizer(tmp_path, use_jamo_alphabet):
    _, obj = _train(tmp_path, f'{use_jamo_alphabet}.json', use_jamo_alphabet=use_jamo_alphabet)

    assert obj['model']['type'] == 'BPE'
    assert len(obj['model']['vocab']) <= VOCAB_SIZE
    assert obj['normalizer'] == {'type': 'NFD'}


def test_normalizer_patch_preserves_content(tmp_path):
    _, plain = _train(tmp_path, 'plain.json')
    _, jamo = _train(tmp_path, 'jamo.json', use_jamo_alphabet=True)

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


def test_invalid_use_jamo_alphabet(tmp_path):
    with pytest.raises(ValueError):
        _train(tmp_path, use_jamo_alphabet='nfc')


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
