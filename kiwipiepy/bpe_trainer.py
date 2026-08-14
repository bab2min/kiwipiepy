'''
.. versionadded:: 0.24.0

한국어 형태소 경계를 고려한 Byte Level BPE 토크나이저를 학습시키는 함수를 제공하는 모듈입니다.
huggingface의 `tokenizers` 라이브러리와 호환되는 BPE 토크나이저를 생성합니다.

또한 `bpe_trainer` 모듈은 간편하게 BPE 토크나이저를 학습할 수 있는 CLI 환경을 제공합니다.

간단 예시:
```bash
python3 -m kiwipiepy.bpe_trainer \\
    some_corpus.txt \\
    my_tokenizer.json \\
    --vocab-size 32000 \\
    --pretokenize j e vcp xsv \\
    --num-threads 8
```

옵션으로 줄 수 있는 인자는 아래와 같습니다.
```text
--vocab-size
    토크나이저의 어휘 집합의 상한치를 지정합니다. (기본값: 32000)

--min-pair-frequency
    BPE merge를 수행할 때 최소 몇 번 이상 등장한 pair를 대상으로 merge를 수행할지 지정합니다. (기본값: 5)

--max-token-length
    토크나이저의 어휘 집합에 포함될 수 있는 토큰의 바이트 상 최대 길이를 지정합니다. (기본값: 30)

--add-prefix-space
    텍스트의 시작 부분에 공백을 추가하여 토크나이저를 학습시킬지 여부를 지정합니다. (기본값: False)

--pretokenize
    pretokenize 단계에서 형태소 경계를 분할할지 여부를 지정합니다. 사용하지 않거나 혹은 다음 값들 중 하나 이상을 선택하여 사용할 수 있습니다.
    j: 조사 경계를 분할합니다.
    e: 어미 경계를 분할합니다.
    vcp: 서술격 조사(긍정 지정사, '-이다') 경계를 분할합니다.
    xsv: 동사/형용사 파생 접미사 경계를 분할합니다.

--use-jamo-alphabet
    설정할 경우 한글의 자음과 모음을 BPE의 초기 알파벳으로 사용합니다. 이 경우 한글 음절은 자모 단위로 분해되어 처리됩니다.
    이렇게 학습된 토크나이저에는 NFD normalizer가 기록되므로, 디코딩 결과는 자모로 분해된 상태로 나옵니다. 음절 형태가 필요하면 NFC 정규화를 적용해야 합니다.

--nfd
    설정할 경우 한글을 제외한 모든 입력을 유니코드 정준 분해(NFD)를 수행한 뒤 BPE를 학습합니다.

--max-digit-length
    숫자 토큰이 가질 수 있는 최대 길이를 지정합니다. 이 길이를 초과하는 숫자 문자열은 pretokenize 단계에서 분할됩니다. (기본값: 3)

--max-repeat-length
    동일한 문자가 반복되는 토큰이 가질 수 있는 최대 길이를 지정합니다. 이 길이를 초과하는 반복 문자열은 pretokenize 단계에서 분할됩니다. (기본값: 8)

--max-whitespace-repeat-length
    동일한 공백 문자가 반복되는 토큰이 가질 수 있는 최대 길이를 지정합니다. 이 길이를 초과하는 반복 문자열은 pretokenize 단계에서 분할됩니다. (기본값: 16)

--num-threads
    학습에 사용할 스레드 수를 지정합니다. 0으로 지정하면 단일 스레드에서 동작하며, -1로 지정하면 시스템의 가용한 모든 스레드를 사용합니다. (기본값: 0)

```
'''

import json
import os
import re
import tempfile
from typing import Callable, List, Optional, Tuple, Union, Iterable, Dict, Any
import unicodedata

from _kiwipiepy import _SwTokenizer

from kiwipiepy import Kiwi


_NFD_NORMALIZER = {'type': 'NFD'}


def nfd_except_hangul(text: str) -> str:
    text = re.sub(r'[^가-힣]+', lambda m: unicodedata.normalize('NFD', m.group()), text)
    return text

def _set_normalizer(save_path: str, normalizer: Optional[Dict[str, Any]]) -> None:

    with open(save_path, encoding='utf-8') as f:
        obj = json.load(f)

    if obj.get('normalizer') == normalizer:
        return
    obj['normalizer'] = normalizer

    directory, name = os.path.split(save_path)
    fd, tmp_path = tempfile.mkstemp(dir=directory or '.', prefix=name + '.', suffix='.tmp')
    try:
        with os.fdopen(fd, 'w', encoding='utf-8') as f:
            json.dump(obj, f, ensure_ascii=False, indent=2, sort_keys=True)
        os.chmod(tmp_path, os.stat(save_path).st_mode & 0o777)
        os.replace(tmp_path, save_path)
    except BaseException:
        try:
            os.remove(tmp_path)
        except OSError:
            pass
        raise

def train_bpe_tokenizer(
    save_path: str,
    texts: Iterable[str],
    vocab_size: int,
    min_pair_frequency: int = 5,
    max_token_length: int = 30,
    add_prefix_space: bool = False,
    pretokenize_j: bool = False,
    pretokenize_e: bool = False,
    pretokenize_vcp: bool = False,
    pretokenize_xsv: bool = False,
    kiwi:Optional[Kiwi] = None,
    use_jamo_alphabet: Union[bool, str] = False,
    max_digit_length: int = 3,
    max_repeat_length: int = 8,
    max_whitespace_repeat_length: int = 16,
    num_workers: int = 0,
    callback: Optional[Callable[[str, int, int], None]] = None,
    show_progress: bool = True,
) -> None:
    '''
Byte Level BPE 토크나이저를 학습시킵니다. pretokenize 과정에서 한국어 형태소 경계를 고려하여 형태소 경계를 넘는 BPE merge를 방지할 수 있는게 특징입니다.

Parameters
----------
save_path : str
    학습된 BPE 토크나이저를 저장할 경로입니다. 확장자는 `.json`으로 지정해야 합니다.
texts : Iterable[str]
    학습에 사용할 텍스트 데이터입니다. 파일 경로를 지정하는 것이 아니라, 텍스트를 직접 담은 iterable 객체를 전달해야 합니다.
vocab_size : int
    토크나이저의 어휘 집합의 상한치를 지정합니다.
min_pair_frequency : int, optional (default: 5)
    BPE merge를 수행할 때 최소 몇 번 이상 등장한 pair를 대상으로 merge를 수행할지 지정합니다.
max_token_length : int, optional (default: 30)
    토크나이저의 어휘 집합에 포함될 수 있는 토큰의 바이트 상 최대 길이를 지정합니다.
add_prefix_space : bool, optional (default: False)
    텍스트의 시작 부분에 공백을 추가하여 토크나이저를 학습시킬지 여부를 지정합니다.
pretokenize_j : bool, optional (default: False)
    pretokenize 단계에서 조사 경계를 분할합니다.
pretokenize_e : bool, optional (default: False)
    pretokenize 단계에서 어미 경계를 분할합니다.
pretokenize_vcp : bool, optional (default: False)
    pretokenize 단계에서 서술격 조사(긍정 지정사, '-이다') 경계를 분할합니다.
pretokenize_xsv : bool, optional (default: False)
    pretokenize 단계에서 동사/형용사 파생 접미사 경계를 분할합니다.
kiwi : Optional[Kiwi], optional (default: None)
    pretokenize_j, e, vcp, xsv 중 하나라도 True로 지정한 경우, 형태소 분석을 수행하기 위해 Kiwi 객체를 반드시 전달해야 합니다.
use_jamo_alphabet : bool or str, optional (default: False)
    False, True, 'nfd' 중 하나를 선택할 수 있습니다. 
    True나 'nfd'로 설정할 경우 한글의 자음과 모음을 BPE의 초기 알파벳으로 사용합니다. 이 경우 한글 음절은 자모 단위로 분해되어 처리됩니다.
    'nfd'의 경우 한글을 제외한 모든 입력을 유니코드 정준 분해(NFD)를 수행한 뒤 BPE를 학습합니다.
    True나 'nfd'로 설정하고 학습한 토크나이저는 `decode()` 결과가 자모로 분해된 상태로 나옵니다.
    원래의 음절 형태가 필요하다면 디코딩 결과에 `unicodedata.normalize('NFC', ...)`를 적용하십시오.
max_digit_length : int, optional (default: 3)
    숫자 토큰이 가질 수 있는 최대 길이를 지정합니다. 이 길이를 초과하는 숫자 문자열은 pretokenize 단계에서 분할됩니다.
max_repeat_length : int, optional (default: 8)
    동일한 문자가 반복되는 토큰이 가질 수 있는 최대 길이를 지정합니다. 이 길이를 초과하는 반복 문자열은 pretokenize 단계에서 분할됩니다.
max_whitespace_repeat_length : int, optional (default: 16)
    동일한 공백 문자가 반복되는 토큰이 가질 수 있는 최대 길이를 지정합니다. 이 길이를 초과하는 반복 문자열은 pretokenize 단계에서 분할됩니다.
num_workers : int, optional (default: 0)
    학습에 사용할 스레드 수를 지정합니다. 0으로 지정하면 단일 스레드에서 동작하며, -1로 지정하면 시스템의 가용한 모든 스레드를 사용합니다.
callback : Optional[Callable[[str, int, int], None]], optional (default: None)
    학습 진행 상황을 추적하기 위한 콜백 함수입니다. 
    콜백함수의 첫번째 인자로는 현재 진행 상황을 나타내는 문자열로 'pretokenizeBegin', 'pretokenizeProgress', 'pretokenizeEnd', 'mergeBegin', 'mergeProgress', 'mergeEnd' 중 하나가 전달됩니다.
    두번째 인자는 현재 진행 step을 나타내는 정수 값이며, 세번째 인자는 total step입니다.
show_progress : bool, optional (default: True)
    학습 진행 상황을 콘솔에 표시할지 여부를 지정합니다. tqdm 라이브러리를 사용하여 진행 상황을 표시합니다.
    '''
    
    if (pretokenize_j or pretokenize_e or pretokenize_vcp or pretokenize_xsv) and kiwi is None:
        raise ValueError("`kiwi` must be specified if any of `pretokenize_j`, `pretokenize_e`, `pretokenize_vcp`, or `pretokenize_xsv` is True.")

    if use_jamo_alphabet not in (False, True, 'nfd'):
        raise ValueError("`use_jamo_alphabet` must be one of False, True, or 'nfd'.")

    if show_progress:
        from tqdm import tqdm
        progress_bar = None

    def _callback(event, current, total):
        if show_progress:
            nonlocal progress_bar
            if event == 'pretokenizeBegin':
                progress_bar = tqdm(total=(total or None), desc="Pretokenize")
            elif event == 'pretokenizeProgress':
                progress_bar.update(current - progress_bar.n)
            elif event == 'pretokenizeEnd':
                progress_bar.update(total - progress_bar.n)
                progress_bar.close()
            elif event == 'mergeBegin':
                progress_bar = tqdm(total=(total or None), desc="Merge")
            elif event == 'mergeProgress':
                progress_bar.update(current - progress_bar.n)
            elif event == 'mergeEnd':
                progress_bar.update(total - progress_bar.n)
                progress_bar.close()

        if callback:
            callback(event, current, total)

    if use_jamo_alphabet == 'nfd':
        texts = map(nfd_except_hangul, texts)

    _SwTokenizer._train_bpe_tokenizer(
        save_path,
        texts,
        vocab_size,
        min_pair_frequency,
        max_token_length,
        add_prefix_space,
        pretokenize_j,
        pretokenize_e,
        pretokenize_vcp,
        pretokenize_xsv,
        kiwi,
        bool(use_jamo_alphabet),
        max_digit_length,
        max_repeat_length,
        max_whitespace_repeat_length,
        num_workers,
        _callback,
    )

    if use_jamo_alphabet:
        _set_normalizer(save_path, _NFD_NORMALIZER)

def _main(args):
    def _data_feeder():
        for input_file in args.input_files:
            print(f"Reading data from: {input_file}")
            yield from open(input_file, 'r', encoding='utf-8')

    print("Training BPE tokenizer with the following parameters:")
    print(f"  Input files: {args.input_files}")
    print(f"  Save path: {args.save_path}")
    print(f"  Vocabulary size: {args.vocab_size}")
    print(f"  Minimum pair frequency: {args.min_pair_frequency}")
    print(f"  Maximum token length: {args.max_token_length}")
    print(f"  Add prefix space: {args.add_prefix_space}")
    print(f"  Use Jamo alphabet: {args.use_jamo_alphabet}")
    print(f"  Maximum digit length: {args.max_digit_length}")
    print(f"  Maximum repeat length: {args.max_repeat_length}")
    print(f"  Maximum whitespace repeat length: {args.max_whitespace_repeat_length}")
    print(f"  Pretokenization options: {args.pretokenize}")
    print(f"  Number of threads: {args.num_threads}")

    kiwi = None
    if args.pretokenize:
        kiwi = Kiwi()
        print("Initialized Kiwi for pretokenization.")

    train_bpe_tokenizer(
        save_path=args.save_path,
        texts=_data_feeder(),
        vocab_size=args.vocab_size,
        min_pair_frequency=args.min_pair_frequency,
        max_token_length=args.max_token_length,
        add_prefix_space=args.add_prefix_space,
        pretokenize_j='j' in (args.pretokenize or []),
        pretokenize_e='e' in (args.pretokenize or []),
        pretokenize_vcp='vcp' in (args.pretokenize or []),
        pretokenize_xsv='xsv' in (args.pretokenize or []),
        kiwi=kiwi,
        use_jamo_alphabet='nfd' if args.nfd else args.use_jamo_alphabet,
        max_digit_length=args.max_digit_length,
        max_repeat_length=args.max_repeat_length,
        max_whitespace_repeat_length=args.max_whitespace_repeat_length,
        num_workers=args.num_workers,
    )

    print("BPE tokenizer training completed successfully. The tokenizer has been saved to:", args.save_path)
    

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('input_files', nargs='+')
    parser.add_argument('save_path')
    parser.add_argument('--vocab-size', default=32000, type=int)
    parser.add_argument('--min-pair-frequency', default=5, type=int)
    parser.add_argument('--max-token-length', default=30, type=int)
    parser.add_argument('--add-prefix-space', default=False, action='store_true')
    parser.add_argument('--pretokenize', nargs='*', choices=['j', 'e', 'vcp', 'xsv'])
    parser.add_argument('--use-jamo-alphabet', default=False, action='store_true')
    parser.add_argument('--max-digit-length', default=3, type=int)
    parser.add_argument('--max-repeat-length', default=8, type=int)
    parser.add_argument('--max-whitespace-repeat-length', default=16, type=int)
    parser.add_argument('--nfd', default=False, action='store_true', help="Normalize input text to NFD before training")
    parser.add_argument('-t', '--num-workers', default=0, type=int)
    _main(parser.parse_args())
