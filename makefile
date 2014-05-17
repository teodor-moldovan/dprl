all: ft

CMD = python test.py TestsCartpole.test_pp_iter
plots:
	python plots.py
noft: 
	$(CMD)
ft:
	faketime -f '-90d' $(CMD)
