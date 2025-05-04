from kss import split_sentences
from sentence_split import run_evaluate

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('datasets', nargs='+')
    parser.add_argument('--write-result')
    parser.add_argument('--write-err')
    parser.add_argument('--backend', default='pecab', choices=['pecab', 'mecab', 'none'])
    args = parser.parse_args()

    all_system_sents = 0
    all_em = []
    all_f1 = []
    all_norm_f1 = []
    split_sentences("foo-bar", backend=args.backend) # warm-up
    for dataset in args.datasets:
        system_sents, em, f1, norm_f1 = run_evaluate(dataset, lambda text:split_sentences(text, backend=args.backend), args.write_result, args.write_err)
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
