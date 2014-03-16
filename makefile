all: ft

CMD = python test.py TestsUnicycle.test_learning
noft: 
	$(CMD)
ft:
	faketime -f '-80d' $(CMD)
