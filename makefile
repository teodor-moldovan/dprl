all: ft

noft: 
	python test.py
ft:
	faketime -f '-40d' python test.py
