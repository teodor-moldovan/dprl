all: ft

CMD = python test.py TestsSwimmer.test_accs
noft: 
	$(CMD)
ft:
	faketime -f '-80d' $(CMD)
