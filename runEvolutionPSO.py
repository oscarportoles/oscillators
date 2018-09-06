#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  6 02:56:04 2018

@author: oscar
"""

import pygmo as po
import numpy as np
import myUDPkaos
import time


generations = 500
sizePop     = 25
library     = 1

pathsave    = '/Users/p277634/python/kaoModel/optimResult/'
#pathsave    = ''
filenameTXT = 'pso_constrins.txt'
filenameNPZ = 'pso_constrins.npz'


# algorithm
algo   = po.algorithm(po.pso(gen=generations))
algo.set_verbosity(1)

# problem
prob   = po.problem(myUDPkaos.KAOsimpleSimuConstr(lib=library))
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
probE   = popE.problem.extract(type(myUDPkaos.KAOsimpleSimuConstr()))
logged  = probE.get_mylogs()
fitness     = np.array([log[0] for log in logged]).astype(np.float32)
ccoef       = np.array([log[1] for log in logged]).astype(np.float32)
velocity    = np.array([log[2] for log in logged]).astype(np.float32)
kL          = np.array([log[3] for log in logged]).astype(np.float32)
kG          = np.array([log[4] for log in logged]).astype(np.float32)
orderG      = np.array([log[5] for log in logged]).astype(np.float32)
orderGsd    = np.array([log[6] for log in logged]).astype(np.float32)
orderL      = np.array([log[7] for log in logged]).astype(np.float32)
orderLsd    = np.array([log[8] for log in logged]).astype(np.float32)
order1      = np.array([log[9] for log in logged]).astype(np.float32)
# get parameters of the model
parametKAO = {"lowBound":probE.lowBound, "upBound":probE.upBound ,"tMax":probE.tMax,
              "tMin":probE.tMin, "fs":probE.fs, "omega":probE.omega, "dlt":probE.dlt,
              "nameDTI":probE.nameDTI ,"nameFC":probE.nameFC ,"dt":probE.dt ,
              "iniCond":probE.iniCond ,"C":probE.C ,"fBands":probE.fBands ,
              "D":probE.D ,"anato":probE.anato ,"empiFC":probE.empiFC ,"lib":probE.lib}
#save file
outfile  = pathsave + filenameNPZ
np.savez(outfile, fitness=fitness, velocity=velocity, kL=kL, kG=kG, Kordr1=order1,
         KordrL=orderL, KordrG=orderG, KordrLsd=orderLsd, KordrGstd=orderGsd,
         parametKAO=parametKAO)

plotOrderKLKGvel(velocity, kL, kG, fitness, orderG, orderL, save=True, filenameNPZ[:-4])