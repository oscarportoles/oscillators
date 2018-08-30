#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul  3 16:55:24 2018

@author: p277634
"""

from platypus import NSGAII, Problem, Real
import pygmo as po
import numpy as np
import myUDPkaos
import time


generations = 3
sizePop     = 3
library     = 2

pathsave    = '/home/oscar/Documents/PythonProjects/kuramotoAO/optimizationResults/'
#pathsave    = '/Users/p277634/python/kaoModel/optimResult/'
filenameTXT = 'ihsTest.txt'
filenameNPZ = 'ihsTest.npz'

print('Running: ', filenameNPZ[:-4])

# Cost function definiton
kaom = myUDPkaos.kaoSimplMultiObjContr(lib=library)
lowBound, upBound = kaom.get_bounds()

# OPtimization Prblem definition
problem = Problem(len(upBound), kaom.get_nobj(), kaom.get_nic())  # (number of decision variables, number of objectives, number of constrains)
problem.nvars = sizePop
problem.types[:] = [Real(lb, ub) for lb,ub in zip(lowBound,upBound)]
problem.constraints[:] = "<=0"
problem.function = kaom.fitness

# Evolutionary algorithm
algorithm = NSGAII(problem)
start  = time.time()
algorithm.run(generations)
print('time evolution: ',time.time()-start)


logged = kaom.get_mylogs()

