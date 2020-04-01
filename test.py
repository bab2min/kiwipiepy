from kiwipiepy import Kiwi
from konlpy.corpus import kolaw

class IOHandler:
    def __init__(self, input, output=None):
        self.input = input
        self.output = output

    def read(self, sent_id):
        if sent_id == 0:
            self.input.seek(0)
            self.iter = iter(self.input)
        try:
            return next(self.iter)
        except StopIteration:
            return None

    def write(self, sent_id, res):
        self.output.write(' '.join(map(lambda x:x[0]+'/'+x[1], res[0][0])) + '\n')

    def __del__(self):
        self.input.close()
        if self.output: self.output.close()

def test_analyze_single():
    kiwi = Kiwi()
    kiwi.prepare()
    for line in kolaw.open('constitution.txt'):
        kiwi.analyze(line)

def test_analyze_multi():
    kiwi = Kiwi()
    kiwi.prepare()
    handle = IOHandler(kolaw.open('constitution.txt'), open('result.txt', 'w', encoding='utf-8'))
    kiwi.analyze(handle.read, handle.write)

def test_async_analyze():
    kiwi = Kiwi()
    kiwi.prepare()
    ret = []
    for line in kolaw.open('constitution.txt'):
        ret.append(kiwi.async_analyze(line))
    ret = [r() for r in ret]

def test_extract_words():
    kiwi = Kiwi()
    kiwi.prepare()
    handle = IOHandler(kolaw.open('constitution.txt'))
    kiwi.extract_words(handle.read)
