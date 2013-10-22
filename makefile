all: ft

noft: 
	python test.py
ft:
	faketime -f '-20d' python test.py
