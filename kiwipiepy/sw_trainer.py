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
        strict_reduction=args.strict_reduction,
        remove_repetitive=args.remove_repetitive,
        iterations=args.iterations,
        reduction_ratio=args.reduction_ratio,
        num_workers=args.num_workers,
        **config.__dict__
    )
    for k, v in p.items():
        print(f"|  {k}: {v!r}")
    print()
    SwTokenizer.train(
        args.save_path,
        MultipleFileLoader(args.input_files),
        config=config,
        vocab_size=args.vocab_size,
        chr_coverage=args.chr_coverage,
        prefix_min_cnt=args.prefix_min_cnt,
        prefix_max_length=args.prefix_max_length,
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
    parser.add_argument('--strict_reduction', default=False, type=_bool)
    parser.add_argument('--remove_repetitive', default=True, type=_bool)
    parser.add_argument('--iterations', default=100, type=int)
    parser.add_argument('--reduction_ratio', default=0.1, type=float)
    parser.add_argument('--num_workers', default=0, type=int)
    main(parser.parse_args())
