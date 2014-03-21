all: ft

CMD = python test.py TestsSwimmer.test_learning
noft: 
	$(CMD)
ft:
	faketime -f '-80d' $(CMD)
