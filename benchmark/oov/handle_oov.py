import re
import os
import sys
import traceback
from collections import Counter

class Model:
    @staticmethod
    def from_name(name, kiwi_model_path=None, bareun_api_key=None):
        if name == 'kiwi': return KiwiModel(kiwi_model_path)
        if name == 'kiwi-largest': return KiwiModel(kiwi_model_path, 'largest')
        if name == 'kiwi-cong': return KiwiModel(kiwi_model_path, 'cong')
        if name == 'kiwi-cong-global': return KiwiModel(kiwi_model_path, 'cong-global')
        if name == 'komoran': return KomoranModel()
        if name == 'kkma': return KkmaModel()
        if name == 'hannanum': return HannanumModel()
        if name == 'mecab': return MecabModel()
        if name == 'okt': return OktModel()
        if name == 'khaiii': return KhaiiiModel()
        if name == 'bareun': return BareunModel(bareun_api_key)
        raise ValueError(f'Unknown model name: {name}')

    def _convert(self, morph):
        return morph
    
    def _tokenize(self, text):
        raise NotImplementedError()
    
    def _is_noun(self, tag):
        return tag.startswith('NN')

    def tokenize(self, text):
        return list(map(self._convert, self._tokenize(text)))
    
    def nouns(self, text):
        return [form for form, tag in self.tokenize(text) if self._is_noun(tag)]

class KiwiModel(Model):
    
    def __init__(self, model_path=None, model_type='none', **kwargs):
        import kiwipiepy
        from kiwipiepy import Kiwi
        print("Initialize kiwipiepy ({})".format(kiwipiepy.__version__), file=sys.stderr)
        self._mdl = Kiwi(model_path=model_path, model_type=model_type, **kwargs)
        self.oov_handling = None
        self.config = None
    
    def _convert(self, morph):
        return morph.form, morph.tag

    def _tokenize(self, text):
        return self._mdl.tokenize(text, oov_handling=self.oov_handling, override_config=self.config)

class KomoranModel(Model):
    def __init__(self):
        import konlpy
        from konlpy import tag
        print("Initialize Komoran from konlpy ({})".format(konlpy.__version__), file=sys.stderr)
        self._mdl = tag.Komoran()
    
    def _tokenize(self, text):
        try:
            return self._mdl.pos(text)
        except:
            return []

class KkmaModel(Model):
    def __init__(self):
        import konlpy
        from konlpy import tag
        print("Initialize Kkma from konlpy ({})".format(konlpy.__version__), file=sys.stderr)
        self._mdl = tag.Kkma()
    
    def _tokenize(self, text):
        try:
            return self._mdl.pos(text)
        except:
            return []

class MecabModel(Model):
    def __init__(self):
        import konlpy
        from konlpy import tag
        print("Initialize Mecab from konlpy ({})".format(konlpy.__version__), file=sys.stderr)
        self._mdl = tag.Mecab()
    
    def _tokenize(self, text):
        try:
            return self._mdl.pos(text, split_inflect=True)
        except TypeError:
            return self._mdl.pos(text)

class HannanumModel(Model):

    def __init__(self):
        import konlpy
        from konlpy import tag
        print("Initialize Hannanum from konlpy ({})".format(konlpy.__version__), file=sys.stderr)
        self._mdl = tag.Hannanum()

    def _convert(self, morph):
        if morph[1] == 'P':
            return morph[0], 'VV'
        return morph[0], morph[1]

    def _is_noun(self, tag):
        return tag == 'N'

    def _tokenize(self, text):
        return self._mdl.pos(text)

class OktModel(Model):

    def __init__(self):
        import konlpy
        from konlpy import tag
        print("Initialize Okt from konlpy ({})".format(konlpy.__version__), file=sys.stderr)
        self._mdl = tag.Okt()
    
    def _convert(self, morph):
        if morph[1] == 'Verb':
            return morph[0][:-1], 'VV'
        return morph[0], morph[1]
    
    def _is_noun(self, tag):
        return tag in ('Noun', 'Foreign')

    def _tokenize(self, text):
        return self._mdl.pos(text, stem=True)

class KhaiiiModel(Model):
    def __init__(self):
        from khaiii import KhaiiiApi
        self._mdl = KhaiiiApi()
        print("Initialize khaiii ({})".format(self._mdl.version()), file=sys.stderr)

    def _tokenize(self, text):
        return [(morph.lex, morph.tag) for word in self._mdl.analyze(text) for morph in word.morphs]

class BareunModel(Model):
    def __init__(self, api_key, host='localhost', port=5656) -> None:
        import bareunpy as brn
        self._mdl = brn.Tagger(api_key, host, port)
        print(f"Initialize Bareun from bareunpy (version={brn.version}, bareun_version={brn.bareun_version})", file=sys.stderr)

    def _tokenize(self, text):
        return self._mdl.tag(text).pos()

def load_dataset(path):
    tag_pattern = re.compile(r'<n(?: e="([^"]+)")?>(.*?)</n>')
    
    ret = []
    for line in open(path, encoding='utf-8'):
        golds = Counter()
        gold_types = {}
        for m in tag_pattern.finditer(line):
            form = m.group(2)
            etype = m.group(1)
            gold_types[form] = etype
            golds[form] += 1
        
        exam = tag_pattern.sub(r'\2', line).rstrip()
        ret.append((golds, gold_types, exam))
    return ret

def evaluate(dataset, model, score_by_type=False, result_output=None):
    gold_per_type = Counter()
    pred_per_type = Counter()
    correct_per_type = Counter()
    gold_chr_per_type = Counter()
    pred_chr_per_type = Counter()
    correct_chr_per_type = Counter()

    results = []
    for golds, gold_types, exam in dataset:
        result = model.nouns(exam)
        if result_output is not None:
            tokens = model.tokenize(exam)
            print(' '.join(f'{form}/{t}' for form, t in gold_types.items() if t), 
                  exam, 
                  ' '.join(f'{form}/{tag}' for form, tag in tokens), sep='\t', file=result_output, flush=True)
        preds = Counter(r.replace(' ', '') for r in result)

        for form, count in golds.items():
            gold_per_type[gold_types.get(form)] += count
            gold_chr_per_type[gold_types.get(form)] += len(form) * count
        
        for form, count in preds.items():
            pred_per_type[gold_types.get(form)] += count
            pred_chr_per_type[gold_types.get(form)] += len(form) * count

        for form, count in (golds & preds).items():
            correct_per_type[gold_types.get(form)] += count
            correct_chr_per_type[gold_types.get(form)] += len(form) * count

        results.append(result)

    all_types = sorted(filter(None, gold_per_type))

    scores = {}
    scores['labeled_recall'] = sum(correct_per_type[t] for t in all_types) / max(sum(gold_per_type[t] for t in all_types), 1)
    if score_by_type:
        scores.update({f'labeled_recall {t}': correct_per_type[t] / max(gold_per_type[t], 1) for t in all_types})

    scores['precision'] = p = sum(correct_per_type.values()) / max(sum(pred_per_type.values()), 1)
    scores['recall'] = r = sum(correct_per_type.values()) / max(sum(gold_per_type.values()), 1)
    scores['f1'] = 2 * p * r / max(p + r, 1)
    if score_by_type:
        for t in all_types:
            scores[f'precision {t}'] = p = correct_per_type[t] / max(pred_per_type[t], 1)
            scores[f'recall {t}'] = r = correct_per_type[t] / max(gold_per_type[t], 1)
            scores[f'f1 {t}'] = 2 * p * r / max(p + r, 1)
    
    scores['chr_precision'] = p = sum(correct_chr_per_type.values()) / max(sum(pred_chr_per_type.values()), 1)
    scores['chr_recall'] = r = sum(correct_chr_per_type.values()) / max(sum(gold_chr_per_type.values()), 1)
    scores['chr_f1'] = 2 * p * r / max(p + r, 1)
    if score_by_type:
        for t in all_types:
            scores[f'chr_precision {t}'] = p = correct_chr_per_type[t] / max(pred_chr_per_type[t], 1)
            scores[f'chr_recall {t}'] = r = correct_chr_per_type[t] / max(gold_chr_per_type[t], 1)
            scores[f'chr_f1 {t}'] = 2 * p * r / max(p + r, 1)
    return scores, results

def test_kiwi_oov_handling(args):
    from kiwipiepy import KiwiConfig
    model = KiwiModel(model_path=args.kiwi_model_path, load_default_dict=args.test_kiwi_with_dictionary, load_multi_dict=args.test_kiwi_with_dictionary)

    settings = [
        *[(f'rule (bias={bias})', {'oov_handling': 'rule', 'config': KiwiConfig(oov_rule_bias=bias)}) for bias in range(-3, 4)],
        *[(f'c (bias={bias})', {'oov_handling': 'chr', 'config': KiwiConfig(oov_chr_bias=bias)}) for bias in range(-3, 4)],
        *[(f'cf (bias={bias}, global_weight=35)', {'oov_handling': 'chr_freq', 'config': KiwiConfig(oov_chr_bias=bias, oov_global_weight=35)}) for bias in range(-3, 4)],
        *[(f'cf (bias=-3, global_weight={w})', {'oov_handling': 'chr_freq', 'config': KiwiConfig(oov_chr_bias=-3, oov_global_weight=w)}) for w in range(5, 105, 5)],
    ]

    if args.result_output:
        result_outputs = [open(f'{os.path.splitext(args.result_output)[0]}_{s[0]}{os.path.splitext(args.result_output)[1]}', 'w', encoding='utf-8') for s in settings]

    print('', '', *[s[0] for s in settings], sep='\t')
    for dataset in args.datasets:
        ds = load_dataset(dataset)
        all_scores = []
        all_results = [ds]
        for i, (name, params) in enumerate(settings):
            model.oov_handling = params['oov_handling']
            model.config = params.get('config')
            score, results = evaluate(ds, model, score_by_type=args.score_by_type, result_output=result_outputs[i] if args.result_output else None)
            all_scores.append(score)
            all_results.append(results)

        for key in score:
            print(os.path.basename(dataset), f'({key})', *((f'{s[key]:.4f}' if s[key] is not None else '-') for s in all_scores), sep='\t')
    
    if args.result_output:
        for f in result_outputs:
            f.close()

def main(args):
    if args.test_kiwi_oov_handling:
        return test_kiwi_oov_handling(args)
    
    models = [Model.from_name(n, 
                              kiwi_model_path=args.kiwi_model_path, 
                              bareun_api_key=args.bareun_api_key) for n in args.target]

    if args.result_output:
        if len(models) == 1:
            result_outputs = [open(args.result_output, 'w', encoding='utf-8')]
        else:
            result_outputs = [open(f'{os.path.splitext(args.result_output)[0]}_{n}{os.path.splitext(args.result_output)[1]}', 'w', encoding='utf-8') for n in args.target]

    print('', '', *args.target, sep='\t')
    for dataset in args.datasets:
        ds = load_dataset(dataset)
        all_scores = []
        all_results = [ds]
        for i, model in enumerate(models):
            score, results = evaluate(ds, model, score_by_type=args.score_by_type, result_output=result_outputs[i] if args.result_output else None)
            all_scores.append(score)
            all_results.append(results)

        for key in score:
            print(os.path.basename(dataset), f'({key})', *((f'{s[key]:.4f}' if s[key] is not None else '-') for s in all_scores), sep='\t')
    
    if args.result_output:
        for f in result_outputs:
            f.close()
    
if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('datasets', nargs='+')
    parser.add_argument('--target', default=['kiwi'], nargs='+', choices=['kiwi', 'kiwi-largest', 'komoran', 'mecab', 'kkma', 'hannanum', 'okt', 'khaiii', 'bareun'])
    parser.add_argument('--kiwi-model-path')
    parser.add_argument('--bareun-api-key')
    parser.add_argument('--test-kiwi-oov-handling', action='store_true')
    parser.add_argument('--test-kiwi-with-dictionary', action='store_true')
    parser.add_argument('--score-by-type', action='store_true')
    parser.add_argument('--result-output')
    main(parser.parse_args())
