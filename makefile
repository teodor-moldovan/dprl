all: ft

noft: 
	python test.py
ft:
	faketime -f '-80d' python test.py
