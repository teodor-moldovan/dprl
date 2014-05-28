all: ft

#CMD = python test.py TestsUnicycle.test_mm_learning
CMD = python test.py TestsAutorotation.test_accs
plots:
	python plots.py
noft: 
	$(CMD)
ft:
	faketime -f '-90d' $(CMD)
