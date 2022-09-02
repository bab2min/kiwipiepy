import sys
import re
import itertools
from difflib import SequenceMatcher
from collections import Counter

import kiwipiepy

space_pat = re.compile(r'\s+')

def do_diff(raw, restored, ignore_whitespace=False, context=0):
    if ignore_whitespace:
        raw = space_pat.sub('', raw)
        restored = space_pat.sub('', restored)
    diff = SequenceMatcher(a=raw, b=restored, autojunk=False)
    
    correct = 0
    for op, s, e, _, _ in diff.get_opcodes():
        if op == 'equal': correct += e - s

    def _gen():
        for op, a_s, a_e, b_s, b_e in diff.get_opcodes():
            if op == 'equal': continue
            diff_a = diff.a[a_s:a_e].strip()
            diff_b = diff.b[b_s:b_e].strip()
            if diff_a == diff_b: continue
            yield diff_a, diff_b, diff.a[a_s-context:a_e+context], diff.b[b_s-context:b_e+context]
    
    return (correct, len(raw), len(restored)), _gen()

def main(args):
    if args.a: args.a = args.a.strip()
    if args.b: args.b = args.b.strip()

    kiwi = kiwipiepy.Kiwi(model_path=args.model_path, num_workers=args.num_workers)
    if args.inputs:
        inputs = itertools.chain.from_iterable(open(i, encoding='utf-8') for i in args.inputs)
    else:
        inputs = sys.stdin
    
    correct, inp, outp = 0, 0, 0
    b_correct, b_outp = 0, 0
    if args.show_count:
        cnt = Counter()
    for tokens, raw in kiwi.tokenize(inputs, echo=True):
        restored = kiwi.join(tokens)
        (a, b, c), d = do_diff(raw, restored, ignore_whitespace=args.ignore_whitespace, context=args.show_context)
        correct += a
        inp += b
        outp += c
        if args.baseline:
            restored = ' '.join(t.form for t in tokens)
            (a, _, c), _ = do_diff(raw, restored, ignore_whitespace=args.ignore_whitespace)
            b_correct += a
            b_outp += c

        if args.show_count:
            cnt.update((a, b) for a, b, *_ in d)
        elif args.show_context:
            for a, b, inp, outp in d:
                if args.a is not None and a != args.a: continue
                if args.b is not None and b != args.b: continue
                if args.show_context:
                    print(inp.strip(), outp.strip(), sep='\t')
                else:
                    print(a, b, sep='\t')
    if args.show_count:
        for (a, b), v in cnt.most_common():
            print(a, b, v, sep='\t')

    if args.baseline:
        try:
            f1 = 2 * b_correct / (inp + b_outp)
        except ZeroDivisionError:
            f1 = float('nan')
        print(f"Baseline ChrF1: {f1:.4f} (A={inp}, B={b_outp}, C={b_correct})")

    try:
        f1 = 2 * correct / (inp + outp)
    except ZeroDivisionError:
        f1 = float('nan')
    print(f"ChrF1: {f1:.4f} (A={inp}, B={outp}, C={correct})")

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('inputs', nargs='*')
    parser.add_argument('--model_path')
    parser.add_argument('--num_workers', default=0, type=int)
    parser.add_argument('-s', '--ignore_whitespace', default=False, action='store_true')
    parser.add_argument('--baseline', default=False, action='store_true')
    parser.add_argument('-c', '--show_count', default=False, action='store_true', help='show counts of the most common errors')
    parser.add_argument('-x', '--show_context', default=0, type=int, help='show context of the errors')
    parser.add_argument('-a', help='filter for a')
    parser.add_argument('-b', help='filter for b')
    main(parser.parse_args())
