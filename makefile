all: ft

CMD = python test.py TestsCartpole.test_accs
noft: 
	$(CMD)
ft:
	faketime -f '-80d' $(CMD)
