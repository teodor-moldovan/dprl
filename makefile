all: ft

CMD = python test.py TestsUnicycle.test_pp
plots:
	python plots.py
noft: 
	$(CMD)
ft:
	faketime -f '-80d' $(CMD)
