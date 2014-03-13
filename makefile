all: ft

CMD = python test.py TestsCartDoublePole.test_cost
noft: 
	$(CMD)
ft:
	faketime -f '-80d' $(CMD)
