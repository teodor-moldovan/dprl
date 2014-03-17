all: ft

CMD = python test.py TestsUnicycle.test_accs
noft: 
	$(CMD)
ft:
	faketime -f '-80d' $(CMD)
