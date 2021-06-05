import os
curpath = os.path.dirname(os.path.abspath(__file__))

from kiwipiepy import Kiwi

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
    for line in open(curpath + '/test_corpus/constitution.txt', encoding='utf-8'):
        kiwi.analyze(line)

def test_analyze_multi():
    kiwi = Kiwi()
    kiwi.prepare()
    handle = IOHandler(open(curpath + '/test_corpus/constitution.txt', encoding='utf-8'), open('result.txt', 'w', encoding='utf-8'))
    kiwi.analyze(handle.read, handle.write)

def test_async_analyze():
    kiwi = Kiwi()
    kiwi.prepare()
    ret = []
    for line in open(curpath + '/test_corpus/constitution.txt', encoding='utf-8'):
        ret.append(kiwi.async_analyze(line))
    ret = [r() for r in ret]

def test_extract_words():
    kiwi = Kiwi()
    kiwi.prepare()
    handle = IOHandler(open(curpath + '/test_corpus/constitution.txt', encoding='utf-8'))
    kiwi.extract_words(handle.read)

def test_tweet():
    kiwi = Kiwi()
    kiwi.prepare()
    kiwi.analyze('''#바둑#장기#오목 귀요미#보드판🐥
#어린이임블리의 놀이였는데, 이제는 가물갸물🙄모르겠
장이요~멍이요~ㅎㅎㅎ다시 한 번 재미를 붙여 보까ㅎ
할 일이 태산인데😭, 하고 싶은건 무궁무진🤦‍♀️ 큰 일이다''')

def test_new_analyze_multi():
    kiwi = Kiwi()
    kiwi.prepare()
    for res in kiwi.analyze(open(curpath + '/test_corpus/constitution.txt', encoding='utf-8')):
        pass

def test_bug_33():
    kiwi = Kiwi()
    kiwi.add_user_word('김갑갑', 'NNP')
    kiwi.prepare()

    print(kiwi.analyze("김갑갑 김갑갑 김갑갑"))
