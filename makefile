all: ft

CMD = python test.py TestsUnicycle.test_learning
plots:
	python plots.py
noft: 
	$(CMD)
ft:
	faketime -f '-80d' $(CMD)
