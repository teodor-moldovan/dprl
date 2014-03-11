all: noft

noft: 
	python test.py TestsPendubot.test_discrete_time
ft:
	faketime -f '-80d' python test.py
