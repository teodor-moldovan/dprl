#!/usr/bin/env python

from distutils.core import setup, Extension

module = Extension('_solver',
                           sources=['solver_wrap.c',
                            'cvxgen/ldl.c',
                            'cvxgen/matrix_support.c',
                            'cvxgen/solver.c',
                            'cvxgen/util.c',
                            'cvxgen/testsolver.c',
                            ],
                )

setup (name = 'solver',
       ext_modules = [module],
       py_modules = ["solver"],
       )
