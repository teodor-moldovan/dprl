all: ft

CMD = python test.py TestsUnicycle.test_pp_iter
noft: 
	$(CMD)
ft:
	faketime -f '-80d' $(CMD)
