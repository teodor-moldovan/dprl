all: plots

CMD = python test.py TestsDoublePendulum.test_mm_learning
plots:
	python plots.py
noft: 
	$(CMD)
ft:
	faketime -f '-80d' $(CMD)
