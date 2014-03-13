all: ft

CMD = python test.py TestsCartDoublePole.test_accs
noft: 
	$(CMD)
ft:
	faketime -f '-80d' $(CMD)
