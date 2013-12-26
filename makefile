all: noft

noft: 
	python test.py
ft:
	faketime -f '-60d' python test.py
