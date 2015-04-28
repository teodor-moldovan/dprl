#!/usr/bin/env python

from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext

import numpy


setup(
    cmdclass = {'build_ext': build_ext},
    ext_modules = [Extension("doublependulum_shooting_sqp_solver",
                             sources=["doublependulum_shooting_sqp_solver.pyx", "doublependulum_QP_solver_shooting.c"],
                             include_dirs=[numpy.get_include(), '.'],
                             language="c++",
                             extra_compile_args=['-ldl', '-lrt'])],
)
