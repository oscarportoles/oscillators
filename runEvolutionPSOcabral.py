#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  6 02:56:04 2018

@author: oscar
"""

import pygmo as po
import numpy as np
import myUDP
import time


generations = 400
sizePop     = 15
pathsave    = '/Users/p277634/python/kaoModel/optimResult/'
#pathsave    = ''
filenameTXT = 'pso_cabral.txt'
filenameNPZ = 'pso_cabral.npz'


# algorithm
algo   = po.algorithm(po.pso(gen=generations))
algo.set_verbosity(1)

# problem
prob   = po.problem(myUDP.kMcabral())
# population
pop    = po.population(prob=prob,size=sizePop)

# evolution
start  = time.time()
popE   = algo.evolve(pop)
print('time evolution: ',time.time()-start)


# save TXT fie with general description of the optimization
bestFstr    = 'champion fitness: ' + str(popE.champion_f[0]) + '; best fit possible: -1'
bestChamp   = 'champion decission vector'
bestXstr    = 'velocity: ' + str(popE.champion_x[0]) + ', kL:' + str(popE.champion_x[1]),', kG: ' + str(popE.champion_x[2])
popStr      = popE.__str__()
algoStr     = algo.__str__()
localtime   = time.localtime(time.time())
dateStr     = str(localtime)
with open(pathsave+filenameTXT, "w") as text_file:
    print(dateStr, end='\n\n', file=text_file)
    print(bestFstr, end='\n', file=text_file)  
    print(bestChamp, end='\n', file=text_file)
    print(bestXstr, end='\n\n', file=text_file)  
    print(algoStr, end='\n', file=text_file)
    print(popStr, end='\n', file=text_file)

# Get logs from the algorithm
loguda = algo.extract(po.pso).get_log()
# get best fitness, CR, and F per generation
funEvals= np.array([loguda[i][2] for i in range(len(loguda))])
bestFit = np.array([loguda[i][2] for i in range(len(loguda))])
meanVel = np.array([loguda[i][3] for i in range(len(loguda))])
meanLfit= np.array([loguda[i][4] for i in range(len(loguda))])
# get parameter for the logging variable in problem class
probE   = popE.problem.extract(type(myUDP.kMcabral()))
logged  = probE.get_mylogs()
fitness     = logged[:,0]
velocity    = logged[:,1]
kG          = logged[:,2]
KordG       = logged[:,3]
KordGsd     = logged[:,5]

#save file
outfile  = pathsave + filenameNPZ
np.savez(outfile, fitness=fitness, velocity=velocity, kG=kG,
         KordrG=KordG, KordrGsd=KordGsd)