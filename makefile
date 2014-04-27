all: noft

CMD = python test.py TestsUnicycle.test_pp_bfgs
plots:
	python plots.py
noft: 
	$(CMD)
ft:
	faketime -f '-80d' $(CMD)
