import bareunpy as brn
from sentence_split import run_evaluate

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('datasets', nargs='+')
    parser.add_argument('--write_result')
    parser.add_argument('--write_err')
    parser.add_argument('--api_key', required=True)
    args = parser.parse_args()

    tagger = brn.Tagger(args.api_key, 'localhost', 5656)
    def split_sentences(text):
        sents = tagger.tag(text, auto_split=True).sentences()
        return [s.text.content for s in sents]

    for dataset in args.datasets:
        run_evaluate(dataset, lambda text:split_sentences(text), args.write_result, args.write_err)
