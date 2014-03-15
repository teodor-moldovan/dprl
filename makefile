all: ft

CMD = python test.py TestsCartpole.test_ddp
noft: 
	$(CMD)
ft:
	faketime -f '-80d' $(CMD)
