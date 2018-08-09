#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul  3 16:55:24 2018

@author: p277634
"""
import pygmo as po
import numpy as np
import myUDP
import time


generations = 400
sizePop     = 36
#pathsave    = '/home/oscar/Documents/PythonProjects/kuramotoAO/optimizationResults/'
pathsave    = '/Users/p277634/python/kaoModel/optimResult/'
filenameTXT = 'NSGA2.txt'
filenameNPZ = 'NSGA2.npz'

print('Running: ', filenameNPZ[:-4])

# algorithm
algo   = po.algorithm(po.nsga2(gen=generations))
algo.set_verbosity(1)
# problem
prob   = po.problem(myUDP.kaoSimplMultiObj())
# population
pop    = po.population(prob=prob,size=sizePop)
# evolution
start  = time.time()
popE   = algo.evolve(pop)
print('time evolution: ',time.time()-start)

# save TXT fie with general description of the optimization
bestFstr  = 'ideal found fit: ' + str(po.ideal(popE.get_f()))  + '; best fit possible: -1'
#bestChamp  = 'champion decission vector'
bestXstr  = 'velocity: ' + str(popE.champion_x[0]) + ', kL:' + str(popE.champion_x[1]),', kG: ' + str(popE.champion_x[2])
popStr   = popE.__str__()
algoStr  = algo.__str__()
localtime = time.localtime(time.time())
dateStr  = str(localtime)
with open(pathsave+filenameTXT, "w") as text_file:
    print(dateStr, end='\n\n', file=text_file)
    print(bestFstr, end='\n', file=text_file)  
    #print(bestChamp, end='\n', file=text_file)
    print(bestXstr, end='\n\n', file=text_file)  
    print(algoStr, end='\n', file=text_file)
    print(popStr, end='\n', file=text_file)

# Get logs from the algorithm
#loguda = algo.extract(po.nsga2).get_log()
# get best fitness, CR, and F per generation
#sigma  = np.array([loguda[i][5] for i in range(len(loguda))])
# get parameter for the logging variable in problem class
probE       = popE.problem.extract(type(myUDP.kaoSimplMultiObj()))
logged      = probE.get_mylogs()
fitness     = np.array([run[0] for run in logged])
velocity    = np.array([run[1] for run in logged])
kG          = np.array([run[2] for run in logged])
kL          = np.array([run[3] for run in logged])
KordG       = np.array([run[4] for run in logged])
KordGsd     = np.array([run[5] for run in logged])
KordL       = np.array([run[6] for run in logged])
KordLsd     = np.array([run[7] for run in logged])

#save file
outfile  = pathsave + filenameNPZ
np.savez(outfile, fitness=fitness, velocity=velocity, kL=kL, kG=kG,
         KordrL=KordL, KordrG=KordG, KordrLstd=KordLsd, KordrGstd=KordGsd)