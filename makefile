all: noft

noft: 
	python test.py TestsSwimmer.test_discrete_time
ft:
	faketime -f '-80d' python test.py
