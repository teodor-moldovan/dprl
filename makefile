all: ft

CMD = python test.py TestsSwimmer.test_pp_iter
noft: 
	$(CMD)
ft:
	faketime -f '-80d' $(CMD)
