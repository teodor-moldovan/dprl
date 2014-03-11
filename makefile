all: noft

noft: 
	python test.py TestsCartDoublePole.test_accs 
ft:
	faketime -f '-80d' python test.py
