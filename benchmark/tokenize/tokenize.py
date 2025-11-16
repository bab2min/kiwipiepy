import re
import sys
import os
from collections import defaultdict
from difflib import SequenceMatcher

class Model:
    @staticmethod
    def from_name(name, kiwi_model_path=None, kiwi_enabled_dialects=None, bareun_api_key=None):
        if name == 'kiwi': return KiwiModel(kiwi_model_path, enabled_dialects=kiwi_enabled_dialects)
        if name == 'kiwi-largest': return KiwiModel(kiwi_model_path, 'largest', enabled_dialects=kiwi_enabled_dialects)
        if name == 'komoran': return KomoranModel()
        if name == 'kkma': return KkmaModel()
        if name == 'hannanum': return HannanumModel()
        if name == 'mecab': return MecabModel()
        if name == 'okt': return OktModel()
        if name == 'khaiii': return KhaiiiModel()
        if name == 'bareun': return BareunModel(bareun_api_key)
        raise ValueError(f'Unknown model name: {name}')

    def _convert(self, morph):
        raise NotImplementedError()
    
    def _tokenize(self, text):
        raise NotImplementedError()

    def tokenize(self, text):
        return list(map(self._convert, self._tokenize(text)))

class KiwiModel(Model):    
    def __init__(self, model_path=None, model_type='none', enabled_dialects=None):
        import kiwipiepy
        from kiwipiepy import Kiwi
        print("Initialize kiwipiepy ({})".format(kiwipiepy.__version__), file=sys.stderr)
        self._mdl = Kiwi(model_path=model_path, model_type=model_type, enabled_dialects=enabled_dialects)
        self._allowed_dialects = enabled_dialects
    
    def _convert(self, morph):
        return morph.form, (morph.tag[:2] if morph.tag.startswith('V') else morph.tag[:1])

    def _tokenize(self, text):
        return self._mdl.tokenize(text, allowed_dialects=self._allowed_dialects)

class KomoranModel(Model):
    def __init__(self):
        import konlpy
        from konlpy import tag
        print("Initialize Komoran from konlpy ({})".format(konlpy.__version__), file=sys.stderr)
        self._mdl = tag.Komoran()
    
    def _convert(self, morph):
        return morph[0], (morph[1][:2] if morph[1].startswith('V') else morph[1][:1])

    def _tokenize(self, text):
        return self._mdl.pos(text)

class KkmaModel(Model):
    def __init__(self):
        import konlpy
        from konlpy import tag
        print("Initialize Kkma from konlpy ({})".format(konlpy.__version__), file=sys.stderr)
        self._mdl = tag.Kkma()
    
    def _convert(self, morph):
        return morph[0], (morph[1][:2] if morph[1].startswith('V') else morph[1][:1])

    def _tokenize(self, text):
        return self._mdl.pos(text)

class MecabModel(Model):
    def __init__(self):
        import konlpy
        from konlpy import tag
        print("Initialize Mecab from konlpy ({})".format(konlpy.__version__), file=sys.stderr)
        self._mdl = tag.Mecab()
    
    def _convert(self, morph):
        return morph[0], (morph[1][:2] if morph[1].startswith('V') else morph[1][:1])

    def _tokenize(self, text):
        try:
            return self._mdl.pos(text, split_inflect=True)
        except TypeError:
            return self._mdl.pos(text)

class HannanumModel(Model):
    disambiguate_verb_adj = False

    def __init__(self):
        import konlpy
        from konlpy import tag
        print("Initialize Hannanum from konlpy ({})".format(konlpy.__version__), file=sys.stderr)
        self._mdl = tag.Hannanum()

    def _convert(self, morph):
        if morph[1] == 'P':
            return morph[0], 'VV'
        return morph[0], morph[1][:1]

    def _tokenize(self, text):
        return self._mdl.pos(text)

class OktModel(Model):
    disambiguate_verb_adj = False

    def __init__(self):
        import konlpy
        from konlpy import tag
        print("Initialize Okt from konlpy ({})".format(konlpy.__version__), file=sys.stderr)
        self._mdl = tag.Okt()
    
    def _convert(self, morph):
        if morph[1] == 'Verb':
            return morph[0][:-1], 'VV'
        return morph[0], morph[1][:1]
    
    def _tokenize(self, text):
        return self._mdl.pos(text, stem=True)

class KhaiiiModel(Model):
    def __init__(self):
        from khaiii import KhaiiiApi
        self._mdl = KhaiiiApi()
        print("Initialize khaiii ({})".format(self._mdl.version()), file=sys.stderr)
    
    def _convert(self, morph):
        form, tag = morph
        return form, (tag[:2] if tag.startswith('V') else tag[:1])

    def _tokenize(self, text):
        return [(morph.lex, morph.tag) for word in self._mdl.analyze(text) for morph in word.morphs]

class BareunModel(Model):
    def __init__(self, api_key, host='localhost', port=5656) -> None:
        import bareunpy as brn
        self._mdl = brn.Tagger(api_key, host, port)
        print(f"Initialize Bareun from bareunpy (version={brn.version}, bareun_version={brn.bareun_version})", file=sys.stderr)

    def _convert(self, morph):
        form, tag = morph
        return form, (tag[:2] if tag.startswith('V') else tag[:1])

    def _tokenize(self, text):
        return self._mdl.tag(text).pos()

def load_dataset(path):
    ret = []
    for line in open(path, encoding='utf-8'):
        line = line.rstrip()
        if not line: continue
        try:
            raw, t, *_ = line.split('\t')
        except:
            print(f'Error at {path}: {line}', file=sys.stderr)
            continue
        tokens = []
        for token in t.split(' '):
            if m := re.match(r'^(.*?)(__([0-9]+))?/([-A-Z]+)$', token):
                form = m.group(1)
                sense = m.group(3) or None
                tag = m.group(4)
                tag = (tag[:2] if tag.startswith('V') else tag[:1])
            else:
                raise ValueError(f'Invalid token format: {token} in {path}: {line}')
            tokens.append((form, tag))
        ret.append((raw, tokens))
    return ret

def flatten_morph(form, tag, sense=None):
    if sense:
        return f'{form}__{sense}/{tag}'
    else:
        return f'{form}/{tag}'

def compute_score(gold_tokens, pred_tokens):
    m = SequenceMatcher(a=gold_tokens, b=pred_tokens)
    common = 0
    for tag, s, e, *_ in m.get_opcodes():
        if tag == 'equal':
            common += (e - s)
    
    precision = common / len(pred_tokens) if pred_tokens else 0
    recall = common / len(gold_tokens) if gold_tokens else 0
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    return precision, recall, f1

def evaluate(dataset, model, error_output=None, print_all_results=False):
    scores = []
    results = []
    for raw, gold_tokens in dataset:
        result = model.tokenize(raw)
        precision, recall, f1 = compute_score(gold_tokens, result)
        results.append(result)
        scores.append(f1)
        if (print_all_results or f1 < 1) and error_output is not None:
            print('(Input)', raw.strip(), file=error_output)
            print('(Gold)', *(flatten_morph(*t) for t in gold_tokens), file=error_output)
            print('(Pred)', *(flatten_morph(*t) for t in result), file=error_output)
            print(f'(P/R/F1) {precision:.4f} / {recall:.4f} / {f1:.4f}\n', file=error_output)
    return sum(scores) / len(scores) if scores else None, results, scores

def main(args):
    model_names = args.target.split(',')
    models = [Model.from_name(n, 
                              kiwi_model_path=args.kiwi_model_path,
                              kiwi_enabled_dialects=args.kiwi_enabled_dialects,
                              bareun_api_key=args.bareun_api_key) for n in model_names]

    if args.error_output_dir:
        os.makedirs(args.error_output_dir, exist_ok=True)
        error_outputs = [open(args.error_output_dir + '/' + name + '.error.txt', 'w', encoding='utf-8') for name in model_names]
    else:
        error_outputs = None

    print('', *model_names, sep='\t')
    accumulated_f1 = [0.0] * len(models)
    for dataset in args.datasets:
        ds = load_dataset(dataset)
        f1s = []
        all_results = [ds]
        all_scores = []
        for i, model in enumerate(models):
            macro_f1, results, scores = evaluate(ds, model, error_output=(error_outputs[i] if error_outputs else None), print_all_results=args.print_all_results)
            f1s.append(macro_f1)
            accumulated_f1[i] += (macro_f1 if macro_f1 is not None else 0)
            all_results.append(results)
            all_scores.append(scores)

        print(os.path.basename(dataset), *((f'{s:.4f}' if s is not None else '-') for s in f1s), sep='\t')
    
    print('<Average>', *((f'{(accumulated_f1[i] / len(args.datasets)):.4f}' if accumulated_f1[i] is not None else '-') for i in range(len(models))), sep='\t')
    
    if error_outputs:
        for f in error_outputs: f.close()
    
if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('datasets', nargs='+')
    parser.add_argument('--target', default='kiwi', help='kiwi,kiwi-largest,komoran,mecab,kkma,hannanum,okt,khaiii,bareun')
    parser.add_argument('--error-output-dir')
    parser.add_argument('--print-all-results', default=False, action='store_true')
    parser.add_argument('--kiwi-model-path')
    parser.add_argument('--kiwi-enabled-dialects', default='standard')
    parser.add_argument('--bareun-api-key')
    main(parser.parse_args())
