#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  6 02:56:04 2018

@author: oscar
"""

import pygmo as po
import numpy as np
import myUDPnodes
import time


generations = 500
sizePop     = 200
#pathsave    = '/home/oscar/Documents/PythonProjects/kuramotoAO/optimizationResults/'
pathsave    = '/Users/p277634/python/kaoModel/optimResult/'
filenameTXT = 'sade_i_nodes_noVel.txt'
filenameNPZ = 'sade_i_nodes_noVel.npz'

# algorithm
algo   = po.algorithm(po.sade(gen=generations,
                              variant=5,
                              variant_adptv=2,
                              ftol=1e-3,
                              xtol=1e-3))
algo.set_verbosity(1)

# problem
prob   = po.problem(myUDPnodes.KAOnodes_noVel())
# population
pop    = po.population(prob=prob,size=sizePop)

# evolution
start  = time.time()
popE   = algo.evolve(pop)
print('time evolution: ',time.time()-start)


# save TXT fie with general description of the optimization
bestFstr  = 'champion fitness: ' + str(popE.champion_f[0]) + '; best fit possible: -1'
bestChamp  = 'champion decission vector'
bestXstr  = 'velocity: ' + str(popE.champion_x[0]) + ', kL:' + str(popE.champion_x[1]),', kG: ' + str(popE.champion_x[2])
popStr   = popE.__str__()
algoStr  = algo.__str__()
localtime = time.localtime(time.time())
dateStr  = str(localtime)
with open(pathsave+filenameTXT, "w") as text_file:
    print(dateStr, end='\n\n', file=text_file)
    print(bestFstr, end='\n', file=text_file)  
    print(bestChamp, end='\n', file=text_file)
    print(bestXstr, end='\n\n', file=text_file)  
    print(algoStr, end='\n', file=text_file)
    print(popStr, end='\n', file=text_file)

# Get logs from the algorithm
loguda = algo.extract(po.sade).get_log()
# get best fitness, CR, and F per generation
bestFit = np.array([loguda[i][2] for i in range(len(loguda))])
bestF   = np.array([loguda[i][3] for i in range(len(loguda))])
bestCR  = np.array([loguda[i][4] for i in range(len(loguda))])
# get parameter for the logging variable in problem class
probE       = popE.problem.extract(type(myUDPnodes.KAOnodes_noVel()))
logged      = probE.get_mylogs()
fitness     = logged[:,0]
velocity    = logged[:,1]
KordL       = logged[:,2]
KordG       = logged[:,3]
KordLsd     = logged[:,4]
KordGsd     = logged[:,5]
kG          = logged[:,6]
kL          = logged[:,7:]

#save file
outfile  = pathsave + filenameNPZ
np.savez(outfile, fitness=fitness, velocity=velocity, kL=kL, kG=kG,
         KordrL=KordL, KordrG=KordG, KordrLstd=KordLsd, KordrGstd=KordGsd)
