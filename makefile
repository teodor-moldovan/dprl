all: ft

CMD = python test.py TestsHeli.test_mm_learning
plots:
	python plots.py
noft: 
	$(CMD)
ft:
	faketime -f '-90d' $(CMD)
