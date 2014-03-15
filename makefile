all: ft

CMD = python test.py TestsUnicycle.test_cost
noft: 
	$(CMD)
ft:
	faketime -f '-80d' $(CMD)
