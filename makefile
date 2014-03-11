all: noft

noft: 
	python test.py TestsSwimmer.test_accs 
ft:
	faketime -f '-80d' python test.py
