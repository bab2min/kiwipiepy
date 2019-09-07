from kiwipiepy import *
kiwi = Kiwi()
kiwi.prepare()
while True:
	txt = input('>>')
	res = kiwi.analyze(txt)[0]
	print(res)