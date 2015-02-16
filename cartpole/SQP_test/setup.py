#!/usr/bin/env python

from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext

import numpy


setup(
    cmdclass = {'build_ext': build_ext},
    ext_modules = [Extension("cartpole_sqp_solver",
                             sources=["cartpole_sqp_solver.pyx", "cartpole_QP_solver.c"],
                             include_dirs=[numpy.get_include(), '.'],
                             language="c++",
                             extra_compile_args=['-ldl', '-lrt'])],
)
