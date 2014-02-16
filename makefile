all: noft

noft: 
	python test.py 
ft:
	faketime -f '-80d' python test.py
