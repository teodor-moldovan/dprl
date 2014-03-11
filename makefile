all: noft

noft: 
	python test.py TestsPendubot.test_disp
ft:
	faketime -f '-80d' python test.py
