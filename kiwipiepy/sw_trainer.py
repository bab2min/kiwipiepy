'''
.. versionadded:: 0.15.1

`sw_trainer` 모듈은 간편하게 SwTokenizer를 학습할 수 있는 CLI 환경을 제공합니다.

간단 예시:
```bash
python3 -m kiwipiepy.sw_trainer \\
    some_corpus.txt \\
    --save_path my_tokenizer.json \\
    --vocab_size 32000
```
`some_corpus.txt` 말뭉치 파일을 분석하여 `32000`종류의 vocab을 가지는 SwTokenizer를 학습해 `my_tokenizer.json`으로 저장합니다.

옵션으로 줄 수 있는 인자는 아래와 같습니다.
```text
--lowercase
    True or False (기본값: False)
    토큰화에 앞서 대문자들을 전부 소문자로 정규화합니다.

--split_chinese
    True or False (기본값: True)
    토큰화에 앞서 모든 한자를 한 글자씩 분리합니다.

--whole_word_unk
    True or False (기본값: False)
    어휘 집합에 속하지 않은 글자가 
    하나라도 포함된 어절 전체를 UNK토큰으로 지정합니다.

--split_punct
    True or False (기본값: True)
    토큰화에 앞서 구두점을 분리합니다.

--simple_tag
    True or False (기본값: True)
    세부 품사태그(예: VV, VX, VA) 대신 대표 품사태그(예: V)를 사용합니다.

--split_verb
    True or False (기본값: True)
    서브워드로 분리가능한 동사가 있다면 더 작은 단위로 분리합니다. 
    (예: 뛰어가다 -> 뛰/V 어/E 가/V 다/E)

--split_eomi
    True or False (기본값: True)
    서브워드로 분리가능한 어미가 있다면 더 작은 단위로 분리합니다. 
    (예: 어요 -> 어/E 요/J)

--use_glue_token
    True or False (기본값: True)
    단어를 이어붙이는 특수 토큰인 Glue토큰을 사용합니다.

--fallback_hangul
    True or False (기본값: True)
    토크나이저의 어휘 집합에 속하지 않은 한글이 있다면 
    초성+중성 / 종성의 형태로 분해하여 처리합니다.

--fallback_byte
    True or False (기본값: False)
    토크나이저의 어휘 집합에 속하지 않은 문자가 있다면 
    UTF8 Byte로 변환하여 처리합니다.

--use_newline_token
    True or False (기본값: False)
    줄바꿈 문자를 토큰으로 사용합니다. False일 경우 공백으로 취급하여 무시합니다.
    줄바꿈 문자로는 fallback_byte 토큰의 10번째 바이트 토큰(\\n)을 사용합니다.

--unk_token
--cls_token
--sep_token
--pad_token
--mask_token
--bos_token
--eos_token
    str
    특수 토큰을 지정합니다.

--save_path
    str, 여러 개 지정 가능
    학습된 SwTokenizer의 저장 경로를 설정합니다.
    vocab_size를 여러 개 지정한 경우 각각의 vocab_size별로 save_path를 지정해주어야 합니다.

--vocab_size
    int, 여러 개 지정 가능
    토크나이저의 어휘 집합의 상한치를 지정합니다.

--chr_coverage
    float (기본값: 0.9995)
    어휘 집합의 후보를 구성할 때 말뭉치에 등장한 글자 중 최대 얼마까지를 
    다룰지를 지정합니다. 기본값은 0.9995로, 이 경우 말뭉치에 등장한 글자들 중
    최대 99.95%를 다룰 수 있도록 어휘 집합을 구성합니다. 이는 한자나 이모지, 
    일부 특수기호처럼 종류는 많지만 각각의 등장빈도가 낮은 글자를 배제하여 
    어휘 집합의 크기를 줄이는 데에 유용합니다.

--prefix_min_cnt
    int (기본값: 5)
    어휘 집합의 후보를 구성할 때 최소 몇 번 이상 등장한 접두어를 포함할지 지정합니다.

--prefix_max_length
    int (기본값: 15)
    어휘 집합의 후보를 구성할 때 포함되는 접두어의 최대 길이를 지정합니다.

--prevent_mixed_digit_tokens
    True or False (기본값: True)
    어휘 집합의 후보를 구성할 때 숫자와 다른 문자가 섞인 토큰을 배제합니다.

--strict_reduction
    True or False (기본값: False)
    어휘 집합을 줄여나갈 때 한 번 배제된 후보가 다시는 사용되지 못하도록 엄격하게 설정합니다. 

--remove_repetitive
    True or False (기본값: True)
    어휘 집합을 구성할때 특정 패턴이 여러번 반복되는 접두어(ex: 안녕안녕안녕)를 배제합니다.

--iterations
    int (기본값: iteration)
    어휘 집합을 줄여나가는 과정을 최대 몇 번 반복할지 설정합니다.
    기본값은 1000이지만, 대체로 1000회보다 더 작은 횟수 안에 학습이 완료됩니다.

--reduction_ratio
    float (기본값: 0.1)
    어휘 집합을 줄여나갈 비율을 설정합니다.

--num_workers
    int (기본값: 0)
    멀티스레딩 시 사용할 스레드의 개수입니다.
    0일 경우 현재 시스템의 가용한 스레드를 전부 사용합니다.

```
'''

from kiwipiepy.sw_tokenizer import SwTokenizer, SwTokenizerConfig

class MultipleFileLoader:

    def _count_lines(self, path, chunk_size=65536):
        lines = 0
        with open(path, 'rb') as f:
            while True:
                b = f.read(chunk_size)
                if not b: break
                lines += b.count(b'\n')
        return lines

    def __init__(self, pathes):
        self._pathes = list(pathes)
        self._total_lines = sum(map(self._count_lines, self._pathes))
    
    def __iter__(self):
        for p in self._pathes:
            yield from open(p, encoding='utf-8')
    
    def __len__(self):
        return self._total_lines

def main(args):
    config = SwTokenizerConfig(
        lowercase=args.lowercase,
        split_chinese=args.split_chinese,
        whole_word_unk=args.whole_word_unk,
        split_punct=args.split_punct,
        simple_tag=args.simple_tag,
        split_verb=args.split_verb,
        split_eomi=args.split_eomi,
        use_glue_token=args.use_glue_token,
        fallback_hangul=args.fallback_hangul,
        fallback_byte=args.fallback_byte,

        unk_token=args.unk_token,
        cls_token=args.cls_token,
        sep_token=args.sep_token,
        pad_token=args.pad_token,
        mask_token=args.mask_token,
        bos_token=args.bos_token,
        eos_token=args.eos_token,
    )
    print("Building SwTokenizer...")
    p = dict(
        vocab_size=args.vocab_size,
        chr_coverage=args.chr_coverage,
        prefix_min_cnt=args.prefix_min_cnt,
        prefix_max_length=args.prefix_max_length,
        prevent_mixed_digit_tokens=args.prevent_mixed_digit_tokens,
        strict_reduction=args.strict_reduction,
        remove_repetitive=args.remove_repetitive,
        iterations=args.iterations,
        reduction_ratio=args.reduction_ratio,
        num_workers=args.num_workers,
        **config.__dict__
    )
    for k, v in p.items():
        print(f"|  {k}: {v!r}")
    print(flush=True)
    SwTokenizer.train(
        args.save_path,
        MultipleFileLoader(args.input_files),
        config=config,
        vocab_size=args.vocab_size,
        chr_coverage=args.chr_coverage,
        prefix_min_cnt=args.prefix_min_cnt,
        prefix_max_length=args.prefix_max_length,
        prevent_mixed_digit_tokens=args.prevent_mixed_digit_tokens,
        strict_reduction=args.strict_reduction,
        remove_repetitive=args.remove_repetitive,
        iterations=args.iterations,
        reduction_ratio=args.reduction_ratio,
        num_workers=args.num_workers,
    )
    print(f'Tokenizer was saved at {args.save_path}')

if __name__ == '__main__':
    def _bool(v):
        vl = v.lower()
        if vl in ('true', 't', '1'):
            return True
        elif vl in ('false', 'f', '0'):
            return False
        else:
            raise ValueError(f"Wrong value {v} for bool type argument.")

    import argparse
    parser = argparse.ArgumentParser(
        'python -m kiwpiepy.sw_trainer', 
        description='Kiwi SwTokenizer Trainer'
    )
    parser.add_argument('input_files', nargs='+')
    parser.add_argument('--lowercase', default=False, type=_bool)
    parser.add_argument('--split_chinese', default=True, type=_bool)
    parser.add_argument('--whole_word_unk', default=False, type=_bool)
    parser.add_argument('--split_punct', default=True, type=_bool)
    parser.add_argument('--simple_tag', default=True, type=_bool)
    parser.add_argument('--split_verb', default=True, type=_bool)
    parser.add_argument('--split_eomi', default=True, type=_bool)
    parser.add_argument('--use_glue_token', default=True, type=_bool)
    parser.add_argument('--fallback_hangul', default=True, type=_bool)
    parser.add_argument('--fallback_byte', default=False, type=_bool)
    parser.add_argument('--use_newline_token', default=False, type=_bool)
    parser.add_argument('--unk_token', default="[UNK]")
    parser.add_argument('--cls_token')
    parser.add_argument('--sep_token')
    parser.add_argument('--pad_token')
    parser.add_argument('--mask_token')
    parser.add_argument('--bos_token')
    parser.add_argument('--eos_token')
    
    parser.add_argument('--save_path', required=True, nargs='+')
    parser.add_argument('--vocab_size', required=True, type=int, nargs='+')
    parser.add_argument('--chr_coverage', default=0.9995, type=float)
    parser.add_argument('--prefix_min_cnt', default=5, type=int)
    parser.add_argument('--prefix_max_length', default=15, type=int)
    parser.add_argument('--prevent_mixed_digit_tokens', default=True, type=_bool)
    parser.add_argument('--strict_reduction', default=False, type=_bool)
    parser.add_argument('--remove_repetitive', default=True, type=_bool)
    parser.add_argument('--iterations', default=1000, type=int)
    parser.add_argument('--reduction_ratio', default=0.1, type=float)
    parser.add_argument('--num_workers', default=0, type=int)
    
    main(parser.parse_args())
