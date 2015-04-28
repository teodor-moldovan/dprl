#!/usr/bin/env python

from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext

import numpy


setup(
    cmdclass = {'build_ext': build_ext},
    ext_modules = [Extension("cython_wrappers",
                             sources=["cython_wrappers.pyx"],
                             include_dirs=[numpy.get_include(), '.'],
                             language="c++",
                             extra_compile_args=['-ldl', '-lrt'])],
)
