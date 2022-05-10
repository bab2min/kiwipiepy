import sys
import readline

from kiwipiepy import Kiwi, __version__

def tokenize(args, kiwi:Kiwi):
    try:
        while True:
            txt = input('>>> ')
            for res in kiwi.analyze(txt, args.top_n, normalize_coda=args.normalize_coda):
                print(res)
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

def main(args):
    print("kiwipiepy v{}".format(__version__))

    kiwi = Kiwi(model_path=args.model_path, sbg=args.sbg)
    if args.task == 'tokenize':
        tokenize(args, kiwi)
    elif args.task == 'space':
        space(args, kiwi)
    elif args.task == 'join':
        join(args, kiwi)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path')
    parser.add_argument('--sbg', default=False, action='store_true')
    parser.add_argument('--top_n', default=1, type=int)
    parser.add_argument('--normalize_coda', default=False, action='store_true')
    parser.add_argument('--reset_whitespace', default=False, action='store_true')
    parser.add_argument('--task', default='tokenize', choices=['tokenize', 'space', 'join'])
    
    main(parser.parse_args())
