#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 31 13:24:09 2018

@author: p277634
"""

import numpy as np
import matplotlib.pyplot as pl
from scipy.stats import pearsonr
from myUDP import Testkao

kao     = Testkao()
dts     = [1e-6,1e-5, 5e-5, 1e-4, 2e-4, 2.5e-4, 5e-4]
ordG    = [None] * len(dts)
ordL    = [None] * len(dts)
xcorG   = np.zeros((len(dts),len(dts)))
xcorL   = np.zeros((len(dts),len(dts)))

for i in range(len(dts)):
    kao.dt = dts[i]
    kao.setVariablesModel([15, 500, 3000])
    
    kS_ = np.float32(kao.kS*kao.dt)
    omga_ = np.float32(kao.omega*kao.dt)
    dlt_ = np.float32(kao.dlt * kao.dt)
    
    z = Testkao._KMAOcommu(kao.phi,kao.r,kao.maxDlay,kao.dlayIdx,kao.eta,np.float32(kao.dlt),kao.fs,kao.dt,np.float32(kao.kS),np.float32(kao.omega))
    #z = Testkao._KMAOcommu(kao.phi,kao.r,kao.maxDlay,kao.dlayIdx,kao.eta,dlt_,kao.fs,kao.dt,kS_,omga_)

    ordG[i] = np.abs(np.mean(z, axis = 0 ))
    ordL[i] = np.mean(np.abs(z), axis = 0 )
    
    
    if i > 0:
        for j in range(i):
            xcorG[i,j],p = pearsonr(ordG[i], ordG[j] )
            xcorL[i,j],p = pearsonr(ordL[i], ordL[j] )

pl.figure()
for  i in range(len(dts)):
    pl.plot(ordG[i], label=str(i))
    pl.legend()
pl.show()

#pl.figure()
#for  i in range(len(dts)):
#    pl.plot(ordL[i],label=str(i))
#pl.show()