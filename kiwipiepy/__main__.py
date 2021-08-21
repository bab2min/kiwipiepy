from kiwipiepy import Kiwi, __version__
print("kiwipiepy v{}".format(__version__))
kiwi = Kiwi()
try:
    while True:
        txt = input('>>')
        res = kiwi.analyze(txt)[0]
        print(res)
except EOFError:
    pass