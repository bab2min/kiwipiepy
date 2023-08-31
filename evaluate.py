## 이 코드는 konlpy의 성능 비교 코드를 바탕으로 작성되었습니다.
## https://github.com/konlpy/konlpy/blob/master/docs/morph.py 

from time import time
from functools import partial

from konlpy import tag
from konlpy.corpus import kolaw
from konlpy.utils import pprint
import kiwipiepy

class Kiwi():
    def __init__(self, model_type='knlm'):
        self.model_type = model_type
        self.k = kiwipiepy.Kiwi(model_type=model_type)

    @property
    def name(self):
        return f'Kiwi({self.model_type})'

    def pos(self, text):
        return [(t.form, t.tag) for t in self.k.tokenize(text)]

def get_tagger_name(tagger_inst):
    try:
        return tagger_inst.name
    except:
        return type(tagger_inst).__name__

def measure_time(taggers, mult=6):
    doc = kolaw.open('constitution.txt').read()*6
    doc = doc.replace('\n', ' ')
    data = [['n', 'load'] + [10**i for i in range(mult)]]
    times = [time()]
    for tagger in taggers:
        inst = tagger()
        diffs = [get_tagger_name(inst)]
        inst.pos("가")
        times.append(time())
        diffs.append(times[-1] - times[-2])
        print('%s\t로딩\t%gs' % (get_tagger_name(inst),  diffs[-1]))
        for i in range(mult):
            doclen = 10 ** i
            r = inst.pos(doc[:doclen])
            times.append(time())
            diffs.append(times[-1] - times[-2])
            print('%s\t%d\t%gs\t(Result Len: %d)' % (get_tagger_name(inst), doclen, diffs[-1], len(r)))
            pprint(r[:5])
        data.append(diffs)
        print()
    return data


def measure_accuracy(taggers, text):
    print('\n%s' % text)
    result = []
    for tagger in taggers:
        inst = tagger()
        print(get_tagger_name(inst))
        r = inst.pos(text)
        pprint(r)
        result.append([get_tagger_name(inst)] + list(map(lambda s: ' / '.join(s), r)))
    return result

if __name__=='__main__':
    MULT = 6

    examples = [u'아버지가방에들어가신다',  # 띄어쓰기
            u'나는 밥을 먹는다', u'하늘을 나는 자동차', # 중의성 해소
            u'아이폰 기다리다 지쳐 애플공홈에서 언락폰질러버렸다 6+ 128기가실버ㅋ'] # 속어

    taggers = [Kiwi, partial(Kiwi, model_type='sbg'), tag.Hannanum, tag.Kkma, tag.Komoran, tag.Okt, tag.Mecab]

    # Time
    data = measure_time(taggers, mult=MULT)
    with open('morph.csv', 'w', encoding='utf-8') as f:
        for d in data:
            f.write('\t'.join(map(str, d)) + '\n')

    # Accuracy
    for i, example in enumerate(examples):
        result = measure_accuracy(taggers, example)
        result = list(map(lambda *row: [i or '' for i in row], *result))
        with open('morph-%s.csv' % i, 'w', encoding='utf-8') as f:
            for d in result:
                f.write('\t'.join(map(str, d)) + '\n')
