all: ft

#CMD = python test.py TestsUnicycle.test_learning
#CMD = python test.py TestsAutorotation.test_learning
#CMD = python test.py TestsAutorotation.test_pp_iter
#CMD = python test.py TestsHeli.test_accs
CMD = python test.py TestsPendulum.test_learning
plots:
	python plots.py
noft: 
	$(CMD)
ft:
	faketime -f '-110d' $(CMD)
#faketime -f '-110d'  python test.py TestsAutorotation.test_pp_iter 
