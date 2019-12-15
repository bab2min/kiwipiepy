from kiwipiepy import Kiwi
from konlpy.corpus import kolaw

class IOHandler:
    def __init__(self, input, output=None):
        self.input = input
        self.output = output

    def read(self, id):
        if id == 0:
            self.input.seek(0)
        return self.input.readline()

    def write(self, id, res):
        print('Analyzed %dth row' % id)
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

def test_extract_words():
    kiwi = Kiwi()
    handle = IOHandler(kolaw.open('constitution.txt'))


