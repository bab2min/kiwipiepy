from kiwipiepy import *
kiwi = Kiwi()
kiwi.prepare()
try:
    while True:
        txt = input('>>')
        res = kiwi.analyze(txt)[0]
        print(res)
except EOFError:
    pass