all: ft

CMD = python test.py TestsUnicycle.test_dyn
noft: 
	$(CMD)
ft:
	faketime -f '-80d' $(CMD)
