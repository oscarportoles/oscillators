# -*- coding: utf-8 -*-
"""
Created on Tue May  8 14:58:04 2018

@author: oscar
"""
#from __future__ import division
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as pl
import time
from numba import jit 
from numba import vectorize

def timing(method):
    def timed(*args, **kw):
        ts = time.time()
        result = method(*args, **kw)
        te = time.time()

        if 'log_time' in kw:
            name = kw.get('log_name', method.__name__.upper())
            kw['log_time'][name] = int((te - ts) * 1000)
        else:
            print '%r  %2.2f ms' % \
                  (method.__name__, (te - ts) * 1000)
        return result

    return timed


#@timing
#@profile
#@jit(cache=True)
#@vectorize
def KMcommuOAanatoDelEulerPar(dt,dlayStep,maxDlay,C,dlt,kS,omga,eta,fs,tMax,iniCond,r,phi):
    """ First order Euler ODE solver for Kuramoto model of communities with
    Ott-Antonsen reduction. Equations from "Hierarchical synchrony of phase
    oscillators in modular networks:. 2012. Per Sebastian Skardal & Juan G
    Restrepo. Physical review E 85, 016208
    
    @ dt:     integration step
    @ dlayStep: number of samples backwards for a time delay [C, C], IF zeros(C) -> no delay
    @ maxDlay: longest delay step, IF 0 -> no delay
    @ C:      Number of communities/nodes
    @ dlt:    spread (half width) of Lorentzian/Cauchy distribution
    @ kS:     connectivity/coupling matrix [C, C]
    @ omge:   nodes intrinsic frequency (rad/s)
    @ eta:    proportion of oscillator per node
    @ fs:     output's sampling frequency, If fs == 0 -> don't do downsampling
    @ tMax:   duration of the simulation time [sec]
    @ iniCond:Flag; do initial/history points. 1- same all, 2- uniformly dist., 3-random
     --Ouput
    @ z:      node's order parameter, complex number [nodes, samples]
    
    Author: Oscar Portoles Marin; ALICE, Groningen University. March 2018
    """
        
    # initialize variables
#    r       = np.empty((C, int(tMax/dt + maxDlay))) # node phase parameter [C, Nsamples to integrate]
#    phi     = np.empty((C, int(tMax/dt + maxDlay))) # node phase parameter [C, Nsamples to integrate]
#    if maxDlay:  # do ODE initial conditions
#        # initial conditions as history for the time delays
#        omegaT  = omga*np.arange(0,maxDlay*dt+dt,dt)
#        if iniCond == 1:   #  all nodes with the same initial conditions
#            r[:,0:maxDlay+1]    = 0.1 * np.ones((C,maxDlay+1))
#            phi[:,0:maxDlay+1]  = np.tile(np.remainder(omegaT,2*np.pi),(C,1))
#        elif iniCond == 2:  # all nodes with equaly spaced initial conditions
#            r[:,0:maxDlay+1]    = np.tile(np.linspace(0.1,0.9,C),(maxDlay+1,1)).transpose()
#            phases              = np.linspace(0.01,2*np.pi,C)
#            for ix, phase in enumerate(phases):
#                phi[ix,0:maxDlay+1]  = np.remainder(omegaT + phase ,2*np.pi)
#    else:
#        r[:,1]       = 0.1
#        phi[:,1]     = np.pi
    commu   = np.arange(0,C)
    pi2     = 2 * np.pi
    eta2    = 0.5 * eta
    kSr     = np.empty((C))
    phiDif  = np.empty((C))
    #cp = cProfile.Profile()
    #cp.enable()
    for n in range(maxDlay,r.shape[1]-1): # solve ODE step by step 
        rsum1       = -dlt * r[:,n]
        rpro1       = eta2 * ( 1 - r[:,n]**2 )
        phipro1     = eta2 * (r[:,n]**2 + 1) / r[:,n]
        for s in range(0,C): 
            idD         = commu + ((n-dlayStep[commu,s])-1) * C
            kSr[:,s]    = kS[:,s] * np.take(r,idD)
            phiDif[:,s] = np.take(phi, idD) - phi[s,n]            
            #idD = n-dlayStep[commu,s]      # index of time-delays per node
            #kSr[:,s]    = kS[:,s] * r[commu,idD]
            #phiDif[:,s] = phi[commu, idD] - phi[s,n]
            #for sp in range(0,C):
            #    kSr[sp]     = kS[sp,s] * r[sp,n-dlayStep[sp,s]]
            #    phiDif[sp]  = phi[sp,n-dlayStep[sp,s]] - phi[s,n]
            sumRsp      = np.sum( kSr * np.cos( phiDif ))
            sumPHIsp    = np.sum( kSr * np.sin( phiDif ))
            rdt         = rsum1[s] + rpro1[s] * sumRsp
            phidt       = omga + phipro1[s] * sumPHIsp
        # add differntial step
            r[s,n+1]    = r[s,n] + dt*rdt
            phi[s,n+1]  = np.remainder(phi[s,n] + dt*phidt, pi2)
    #cp.disable()
    #cp.print_stats()
    r       = r[:,maxDlay+1:]     # remove history samples used in the begining
    phi     = phi[:,maxDlay+1:]
    if fs:   # simple downsampling (there may be aliasing)
        r   = r[:,::int(1/(fs*dt))] 
        phi = phi[:,::int(1/(fs*dt))]
    return r * np.exp(1j* phi)
    


iniCond = 2                     # how initial conditions [r, phi] are generated: 1- same all, 2- uniformll dist., 3-random
OnDlay  = True                  # 1:  simulate with time delays, 0: without time delays 
kG      = 25*90.0               # global coupling, couplilng between commiunities.
kL      = 150.0                 # local coupling, coupling inside the commiunities.
dlt     = 1.0                   # delta parameter (spread) for Lorentzian distribution of frequencies.
fr      = 40.0                  # frequency of oscillatros [Hz]
omga    = fr * 2*np.pi          # Omega parameter (mean) for lorentzian distribution of frequencies.
tMax    = 5.0                  # maximum integration time
tMin    = 10.                   # remove the previous simulation time. Only take satable time 
dt      = 1e-4                  # integration time step   
vel     = 7.6                   # transmision velocity of neural impulses [meter/second]
fs      = 500.0                 # sampling frequency [Hz] of the genrated signal
fBands  = np.array([[2.0,    6.0], # frequency bands for the band-pass filter
                    [4.0,    8.0],
                    [6.0,    10.5],
                    [8.0,    13.0],
                    [10.5,   21.5],
                    [13.0,   30.0],
                    [21.5,   39.0],
                    [30.0,   48.0],
                    [39.0,   66.0],
                    [52.0,   80.0]])

pathdata    = '/home/oscar/Documents/MATLAB/Kuramoto/Cabral/AAL_matrices.mat'
anato       = sio.loadmat(pathdata)   # C: structural connectivity, D: distance between areas.
kS          = anato['C']                # Strucural/anatomical network  
D           = anato['D']                # Distances beween nodes
C           = np.shape(kS)[1]           # number of nodes/brain areas
eta         = 1.0 / C;                  # proportion of oscillator per community

# set copling strengt with structural connectivity
kS          = kS / np.mean(kS[~np.identity(C,dtype=bool)])   # normalize structural network to a mean = 1
# kS(isnan(kS)) = 0;  # only using Konig data
# kS          = kS + kS';
kS          = kS * kG / C               # Global coupling
np.fill_diagonal(kS,kL)                 # Local coupling
# if ~issymmetric(kS), error('Adjecency matrix is not symmetric'), end

# set delays
if OnDlay:
    dlay        = D / (1000.0 * vel)                # [seconds] correct from mm to m
    dlayStep    = np.around(dlay / dt).astype(int)  # delay on steps backwards to be done
    maxDlay     = int(np.max(dlayStep))             # number of time steps for the longest delay
else:
    dlayStep    = np.zeros([C,C],int)
    maxDlay     = 0
r       = np.empty((C, int(tMax/dt + maxDlay))) # node phase parameter [C, Nsamples to integrate]
phi     = np.empty((C, int(tMax/dt + maxDlay))) # node phase parameter [C, Nsamples to integrate]
if maxDlay:  # do ODE initial conditions
    # initial conditions as history for the time delays
    omegaT  = omga*np.arange(0,maxDlay*dt+dt,dt)
    if iniCond == 1:   #  all nodes with the same initial conditions
        r[:,0:maxDlay+1]    = 0.1 * np.ones((C,maxDlay+1))
        phi[:,0:maxDlay+1]  = np.tile(np.remainder(omegaT,2*np.pi),(C,1))
    elif iniCond == 2:  # all nodes with equaly spaced initial conditions
        r[:,0:maxDlay+1]    = np.tile(np.linspace(0.1,0.9,C),(maxDlay+1,1)).transpose()
        phases              = np.linspace(0.01,2*np.pi,C)
        for ix, phase in enumerate(phases):
            phi[ix,0:maxDlay+1]  = np.remainder(omegaT + phase ,2*np.pi)
else:
    r[:,1]       = 0.1
    phi[:,1]     = np.pi
start = time.time()
z = KMcommuOAanatoDelEulerPar(dt,dlayStep,maxDlay,C,dlt,kS,omga,eta,fs,tMax,iniCond,r,phi)
end = time.time()

pl.plot(np.imag(np.mean(z,axis=0)),np.real(np.mean(z,axis=0)))
pl.show()
print end - start
