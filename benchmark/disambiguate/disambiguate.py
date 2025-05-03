import sys
import os

class Model:
    disambiguate_verb_adj = True

    @staticmethod
    def from_name(name, kiwi_model_path=None, bareun_api_key=None):
        if name == 'kiwi': return KiwiModel(kiwi_model_path)
        if name == 'kiwi-largest': return KiwiModel(kiwi_model_path, 'largest')
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
    def __init__(self, model_path=None, model_type='knlm'):
        import kiwipiepy
        from kiwipiepy import Kiwi
        print("Initialize kiwipiepy ({})".format(kiwipiepy.__version__), file=sys.stderr)
        self._mdl = Kiwi(model_path=model_path, model_type=model_type)
    
    def _convert(self, morph):
        return morph.form, (morph.tag[:2] if morph.tag.startswith('V') else morph.tag[:1])

    def _tokenize(self, text):
        return self._mdl.tokenize(text)

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
        return self._mdl.pos(text, split_inflect=True)

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
            answer, exam = line.split('\t')
        except:
            print(f'Error at {path}: {line}', file=sys.stderr)
            continue
        form, tag = answer.split('/')
        if tag.startswith('V'): tag = tag[:2]
        else: tag = tag[:1]
        ret.append(((form, tag), exam))
    return ret

def evaluate(dataset, model, error_output=None, print_all_results=False):
    acc, tot = 0, 0
    for answer, exam in dataset:
        if answer[1] == 'VA' and not model.disambiguate_verb_adj:
            return None
        result = model.tokenize(exam)
        correct = answer in set(result)
        acc += int(correct)
        tot += 1
        if (print_all_results or not correct) and error_output is not None:
            print('/'.join(answer), ':', *map('/'.join, result), file=error_output)
    return acc / tot

def main(args):
    model_names = args.target.split(',')
    models = [Model.from_name(n, kiwi_model_path=args.kiwi_model_path, bareun_api_key=args.bareun_api_key) for n in model_names]

    if args.error_output_dir:
        os.makedirs(args.error_output_dir, exist_ok=True)
        error_outputs = [open(args.error_output_dir + '/' + name + '.error.txt', 'w', encoding='utf-8') for name in model_names]
    else:
        error_outputs = None

    if args.write_comparison:
        comp_out = open(args.write_comparison, 'w', encoding='utf-8')

    print('', *model_names, sep='\t')
    for dataset in args.datasets:
        ds = load_dataset(dataset)
        scores = []
        for i, model in enumerate(models):
            acc = evaluate(ds, model, error_output=(error_outputs[i] if error_outputs else None), print_all_results=args.print_all_results)
            scores.append(acc)
        if args.write_comparison:
            print(f'<table>', file=comp_out)
            print(f'<caption>{dataset}</caption>', file=comp_out)
            print(f'<tr><th>입력</th><th>분석기</th><th>출력</th></tr>', file=comp_out)
            for exam, *results in zip(*all_results):
                print(f'<tr><th rowspan="{len(results)}">{exam}</th><th>{model_names[0]}</th><td>{" ".join(flatten_morph(*r) for r in results[0])}</td></tr>', file=comp_out)
                for model_name, result in zip(model_names[1:], results[1:]):
                    print(f'<tr><th>{model_name}</th><td>{" ".join(flatten_morph(*r) for r in result)}</td></tr>', file=comp_out)
            print(f'</table>', file=comp_out)

        print(os.path.basename(dataset), *((f'{s:.4f}' if s is not None else '-') for s in scores), sep='\t')
    
    if error_outputs:
        for f in error_outputs: f.close()
    
    if args.write_comparison:
        comp_out.close()
            
if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('datasets', nargs='+')
    parser.add_argument('--target', default='kiwi', help='kiwi,kiwi-largest,komoran,mecab,kkma,hannanum,okt,khaiii,bareun')
    parser.add_argument('--error-output-dir')
    parser.add_argument('--print-all-results', default=False, action='store_true')
    parser.add_argument('--write-comparison')
    parser.add_argument('--kiwi-model-path')
    parser.add_argument('--bareun-api-key')
    main(parser.parse_args())
