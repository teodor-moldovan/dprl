all: ft

#CMD = python test.py TestsUnicycle.test_learning
#CMD = python test.py TestsAutorotation.test_learning
#CMD = python test.py TestsAutorotation.test_pp_iter
#CMD = python test.py TestsHeli.test_accs
#CMD = python robotarm.py TestsRobotArm7dof.test_accs
CMD = python cartpole.py TestsCartpole.test_accs
#CMD = python robotarm.py TestsRobotArm3dof.test_learning
#CMD = python pendulum.py TestsPendulum.test_accs
plots:
	python plots.py
noft: 
	$(CMD)
ft:
	faketime -f '-150d' $(CMD)
