from koalanlp.Util import initialize, finalize
from koalanlp.proc import SentenceSplitter, Tagger
from koalanlp import API
from sentence_split import run_evaluate

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('datasets', nargs='+')
    parser.add_argument('--write-result')
    parser.add_argument('--write-err')
    parser.add_argument('--backend', default='OKT', choices=['OKT', 'HNN', 'KMR', 'RHINO', 'EUNJEON', 'ARIRANG', 'KKMA'])
    args = parser.parse_args()

    initialize(**{args.backend:'LATEST'})
    if args.backend in ('OKT', 'HNN'):
        splitter = SentenceSplitter(getattr(API, args.backend))
    else:
        tagger = Tagger(getattr(API, args.backend))
        splitter = lambda text: [sent.surfaceString() for sent in tagger(text)]
    
    all_system_sents = 0
    all_em = []
    all_f1 = []
    all_norm_f1 = []

    splitter("foo-bar") # warm-up

    for dataset in args.datasets:
        system_sents, em, f1, norm_f1 = run_evaluate(dataset, splitter, args.write_result, args.write_err)
        all_system_sents += system_sents
        all_em += em
        all_f1 += f1
        all_norm_f1 += norm_f1
    
    print("[Overall]")
    print(
        f"Gold: {len(all_f1)} sents, "
        f"System: {all_system_sents} sents, "
        f"EM: {sum(all_em) / len(all_em):.5f}, "
        f"F1: {sum(all_f1) / len(all_f1):.5f}, "
        f"Normalized F1: {sum(all_norm_f1) / len(all_norm_f1):.5f}"
    )
    
    finalize()