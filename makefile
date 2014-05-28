all: ft

#CMD = python test.py TestsUnicycle.test_mm_learning
CMD = python test.py TestsAutorotation.test_pp_iter
plots:
	python plots.py
noft: 
	$(CMD)
ft:
	faketime -f '-90d' $(CMD)
