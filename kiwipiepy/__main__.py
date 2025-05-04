import sys
from pprint import pprint
try:
    import readline
except:
    pass

from kiwipiepy import Kiwi, __version__

def tokenize(args, kiwi:Kiwi):
    try:
        while True:
            txt = input('>>> ')
            for res in kiwi.analyze(txt, args.top_n, normalize_coda=args.normalize_coda, saisiot=args.saisiot):
                pprint(res)
    except (EOFError, KeyboardInterrupt):
        print()

def space(args, kiwi:Kiwi):
    try:
        while True:
            txt = input('>>> ')
            res = kiwi.space(txt, reset_whitespace=args.reset_whitespace)
            print(res)
    except (EOFError, KeyboardInterrupt):
        print()

def join(args, kiwi:Kiwi):
    try:
        while True:
            try:
                txt = input('>>> ')
                tokens = []
                for t in txt.split():
                    form, tag = t.rsplit('/', 1)
                    tokens.append((form, tag))
                res = kiwi.join(tokens)
                print(res)
            except ValueError:
                print("Wrong Input: ", txt, file=sys.stderr)
    except (EOFError, KeyboardInterrupt):
        print()

def split(args, kiwi:Kiwi):
    try:
        while True:
            txt = input('>>> ')
            for res in kiwi.split_into_sents(txt, normalize_coda=args.normalize_coda, saisiot=args.saisiot):
                pprint(res)
    except (EOFError, KeyboardInterrupt):
        print()

def main(args):
    print("kiwipiepy v{}".format(__version__))

    kiwi = Kiwi(model_path=args.model_path, model_type=args.model_type, typos=args.typos, typo_cost_threshold=args.typo_cost_threshold)
    if args.task == 'tokenize':
        tokenize(args, kiwi)
    elif args.task == 'space':
        space(args, kiwi)
    elif args.task == 'join':
        join(args, kiwi)
    elif args.task == 'split':
        split(args, kiwi)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-path')
    parser.add_argument('--model-type', choices=['none', 'largest', 'knlm', 'sbg', 'cong', 'cong-global'])
    parser.add_argument('--top-n', default=1, type=int)
    parser.add_argument('--normalize-coda', default=False, action='store_true')
    parser.add_argument('--reset-whitespace', default=False, action='store_true')
    parser.add_argument('--task', default='tokenize', choices=['tokenize', 'space', 'join', 'split'])
    parser.add_argument('--typos')
    parser.add_argument('--typo-cost-threshold', default=2.5, type=float)
    parser.add_argument('--saisiot', default=None, action='store_true')
    parser.add_argument('--no-saisiot', action='store_false', dest='saisiot')
    
    main(parser.parse_args())
