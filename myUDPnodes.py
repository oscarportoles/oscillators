#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun  1 15:24:15 2018

@author: oscar
"""
import numpy as np
import scipy.io as sio
from numba import jit, prange
import scipy.signal as sg
from scipy.stats import pearsonr

#from numba import float32, int32, complex128, prange

class KAOnodes():
    def __init__(self):
        self.iniCond  = 1 # Initial conditions - 1: psudo-random, 2: phi uniform, r const(0.3), other: zeros
        self.tMax     = np.float32(4.5)
        self.tMin     = np.float32(1.0)
        self.fs       = np.float32(250.0)
        self.omega    = np.float64(2*np.pi * 40.0)
        self.dlt      = np.float64(0.25)
        self.pathData = '/Users/p277634/python/kaoModel/'
        self.nameDTI  = 'AAL_matrices.mat'         # anatomical network
        self.nameFC   = 'Real_Band_FC.mat'     # Empirical Functional connectivity
        self.dt       = np.float64(5e-4)
        self._getEmpiricalData()
        self._desingFilterBands()
        self.log      = np.empty((0,7 + self.C))
        
    def get_mylogs(self):
        return self.log
    
    def get_name(self):
        return "kao with all nodes"
    
    def get_bounds(self):
        """Boundaries on: velocity, kG, kL"""
        upbound     = [3500] * self.C
        upbound     = [20, 9000] + upbound
        lowbound    = [0.1] * self.C
        lowbound    = [1.0, 0.1] + lowbound
        return (lowbound, upbound)
          
    def _getEmpiricalData(self): 
        # load anatomical data
        loaddata    = self.pathData + self.nameDTI
        dti         = sio.loadmat(loaddata)     # C: structural connectivity, D: distance between areas.
        self.D      = dti['D']                  # Distances beween nodes
        anato       = dti['C']                  # Strucural/anatomical network
        self.C      = np.shape(self.D)[1]       # number of nodes/brain areas
        self.eta    = np.float32(1.0 / self.C)             # proportion of oscillator per community
        #self.anato  = anato / np.mean(anato[~np.identity(self.C,dtype=bool)])   # normalize structural network to a mean = 1
        self.anato  = anato / np.mean(anato)
        # load functional data
        loaddata    = self.pathData + self.nameFC
        empiri      = sio.loadmat(loaddata)              # fBands: frequency bands, FCf:functional connectivity
        self.fBands = empiri['freq_bands'].astype(np.float32) # bandpass filter frequency bands
        self.empiFC      = empiri['FC_Env_mean']              # empiprical functional connectivity
        self.empiProfile = []
        for ix in range(0,self.fBands.shape[0]):         # Profile Empirical data
            empi1            = self.empiFC[ix,...]
            self.empiProfile = np.append(self.empiProfile, empi1[np.triu_indices(self.C,1)])
        self.empiProfile = np.clip(self.empiProfile, a_min=0, a_max=None)
        
    def _desingFilterBands(self):
        nyq   = self.fs / 2.0
        trans = 2.0
        self.coeFil = []
        for freq in self.fBands:
            # Filter frequency bands
            passCut = freq / nyq
            stopCut = [(freq[0] - trans) / nyq, (freq[1] + trans) / nyq]
            self.coeFil.append(sg.iirdesign(passCut, stopCut, gpass=0.0025, gstop=30.0,
                                            analog=False, ftype='cheby2', output='sos'))
        # Filter envelops
        self.coeFilEnv = sg.iirdesign(0.5 / nyq, (0.5+trans)/nyq , gpass=0.0025, gstop=30.0,
                                            analog=False, ftype='cheby2', output='sos')
    
    def _doKuramotoOrder(self, z):
        # global order
        ordG        = np.abs( np.mean( z[:,int(self.tMin*self.fs):], axis = 0 ))
        orderG      = np.mean(ordG)
        orderGstd   = np.std(ordG)
        # local order
        ordL        = np.mean( np.abs(z[:,int(self.tMin*self.fs):]), axis = 0 )
        orderL      = np.mean(ordL)
        orderLstd   = np.std(ordL)
        return orderG, orderGstd, orderL, orderLstd      
    
    def fitness(self,x):
        vel     = x[0]
        kG      = x[1]
        kL      = x[2:]     # Kl is an scalar if kL is fix for all nodes, or kL is an array if kL is free
        kS      = self.getAnatoCoupling(kG,kL)
        dlayStep, maxDlay = self.getDelays(vel)
        r, phi  = self._doNodeContainers(maxDlay)
        dlayIdx = self.doIndexDelay(r,dlayStep)
        
        # scales by dt,  try to reduce floating point error, and speed-up
        kS_ = np.float32(kS*self.dt)
        omga_ = np.float32(self.omega*self.dt)
        dlt_ = np.float32( - self.dlt * self.dt)
        
        z    = KAOnodes._KMAOcommu(phi,r,maxDlay,dlayIdx,self.eta,dlt_,self.fs,self.dt,kS_,omga_)    
        #self.z = z
        fit         = self._fitFilterBands(z)
        orderG, orderGstd, orderL, orderLstd = self._doKuramotoOrder(z)
        self.log    = np.vstack((self.log,
                                 np.append( [fit,vel,orderL,orderG,orderLstd,orderGstd,kG] , kL)))
        return np.array([fit])
    
    def doIndexDelay(self,r,dlayStep):
        nodes   = np.tile(np.arange(self.C),(self.C,1)) * r.shape[1]
        outpu   = nodes - dlayStep
        return outpu
    
    def getAnatoCoupling(self,kG,kL):
        """Get anatomical network with couplings"""
        kS = self.anato * kG / self.C        # Globa  coupling
        np.fill_diagonal(kS,kL)              # Local coupling
        return np.float64(kS)
    
    def getDelays(self,vel):
        """Return maximum delay and delay steps in samples"""
        dlay        = self.D / (1000.0 * vel)                     # [seconds] correct from mm to m
        dlayStep    = np.around(dlay / self.dt).astype(np.int32)  # delay on steps backwards to be done
        maxDlay     = np.int32(np.max(dlayStep))                  # number of time steps for the longest delay
        return dlayStep, maxDlay
    
    def _fitFilterBands(self,z):
        simuProfile = []
        for coefsos in self.coeFil:
            # filter frequency bands
            zFilt   = sg.sosfiltfilt(coefsos, np.imag(z), axis=1, padtype='odd')
            zEnv    = np.abs(sg.hilbert(zFilt, axis=1))
            # filter envelope
            zEnvFilt= sg.sosfiltfilt(self.coeFilEnv, zEnv, axis=1, padtype='odd')
            # Correlation discarding warmup time
            envCo = np.corrcoef(zEnvFilt[:,int(self.tMin*self.fs):-int(self.tMin*self.fs/2)], rowvar=True)
            # set to zero negative correlations
            envCo = np.clip(envCo, a_min=0, a_max=None)
            simuProfile  = np.append(simuProfile, envCo[np.triu_indices(z.shape[0],1)])
        #print(simuProfile.shape)
        ccoef, pval = pearsonr(simuProfile, self.empiProfile)
        return -1 * ccoef
    
    #complex128[:,:](float32[:,:],float32[:,:],int32,int32[:,:],float32,float32,float32,float32,float32[:,:],float32), 
    @jit(nopython=True,cache=True,nogil=True,parallel=True,fastmath=True)
    def _KMAOcommu(phi,r,maxDlay,dlayStep,eta,dlt,fs,dt,kS,omga):
        C       = phi.shape[0]
        pi2     = np.float32(2 * np.pi)
        eta2    = np.float32(0.5 * eta)
        sumRsp  = np.zeros((C),dtype=np.float32)
        sumPHIsp= np.zeros((C),dtype=np.float32)
        for n in range(maxDlay,phi.shape[1]-1):
            rsum1       = dlt * r[:,n]
            rpro1       = eta2 * ( 1 - r[:,n] * r[:,n])
            phipro1     = eta2 * (r[:,n] + 1 / r[:,n])
            idD         = n + dlayStep
            phi_r = phi.ravel()
            r_r   = r.ravel()
            for s in prange(C):
                phiDif      = phi_r[idD[s,:]] - phi[s,n]
                kSr         = kS[:,s] * r_r[idD[s,:]]
                sumRsp[s]   = np.sum( kSr * np.cos( phiDif ))
                sumPHIsp[s] = np.sum( kSr * np.sin( phiDif ))
            rdt         = rsum1 + rpro1 * sumRsp
            phidt       = omga + phipro1 * sumPHIsp
            # add differntial step
            r[:,n+1]    = r[:,n] + rdt
            phi[:,n+1]  = np.remainder(phi[:,n] + phidt, pi2)
        r       = r[:,maxDlay+1:]     # remove history samples used in the begining
        phi     = phi[:,maxDlay+1:]
        # simple downsampling (there may be aliasing)
        r   = r[:,::np.int32(1./(fs*dt))] 
        phi = phi[:,::np.int32(1./(fs*dt))]
        return r * np.exp(1j* phi)
    
    def _doNodeContainers(self,maxDlay):      
        # node's variables
        r           = np.zeros((self.C, int(self.tMax/self.dt + maxDlay)), dtype=np.float32) # node phase parameter [C, Nsamples to integrate]
        phi         = np.zeros((self.C, int(self.tMax/self.dt + maxDlay)), dtype=np.float32) # node phase parameter [C, Nsamples to integrate]
        # initial conditions as history for the time delays
        if self.iniCond == 1:   # random 
            np.random.seed(37)
            phiRa   = np.tile(np.random.uniform(-np.pi,np.pi,self.C), (maxDlay+1,1)).T
            time    = np.tile(np.linspace(0, (maxDlay+1)*self.dt, maxDlay+1, dtype=np.float32),(self.C,1))
            phi[:,0:maxDlay+1]  = np.float32(np.remainder(time * self.omega + phiRa, 2*np.pi))
            np.random.seed(37)
            r[:,0:maxDlay+1]    = np.float32(np.tile(np.random.random(self.C), (maxDlay+1,1))).T
        elif self.iniCond == 2: # phase equaly distributed around the circle 
            omegaT      = self.omega * np.linspace(0, maxDlay*self.dt+self.dt,maxDlay+1, dtype=np.float32)
            r[:,0:maxDlay+1]    = 0.3 * np.ones((self.C,maxDlay+1),dtype=np.float32)
            phi[:,0:maxDlay+1]  = np.tile(np.remainder(omegaT,2*np.pi),(self.C,1))
        return r, phi
    

class KAOnodesMultiObj():
    """ KM model with AO reduction. Optimized delay, global, and local coupling.
        Each frequency band is one cost function, to be solved as multi-objective 
        optimization problem."""
    def __init__(self):
        self.iniCond  = 1 # Initial conditions - 1: psudo-random, 2: phi uniform, r const(0.3), other: zeros
        self.tMax     = np.float32(4.5)
        self.tMin     = np.float32(1.0)
        self.fs       = np.float32(250.0)
        self.omega    = np.float64(2*np.pi * 40.0)
        self.dlt      = np.float64(1.0)
        self.pathData = '/Users/p277634/python/kaoModel/'
        self.nameDTI  = 'AAL_matrices.mat'         # anatomical network
        self.nameFC   = 'Real_Band_FC.mat'     # Empirical Functional connectivity
        self.dt       = np.float64(5e-4)
        self._getEmpiricalData()
        self._desingFilterBands()
        self.log      = []
        
    def get_mylogs(self):
        return self.log
    
    def get_name(self):
        return "kao with all nodes, multiobjective optimization"
    
    def get_nobj(self):
        return self.fBands.shape[0]
    
    
    def get_bounds(self):
        """Boundaries on: velocity, kG, kL"""
        upbound     = [3500] * self.C
        upbound     = [20, 9000] + upbound
        lowbound    = [0.1] * self.C
        lowbound    = [1.0, 0.1] + lowbound
        return (lowbound, upbound)
          
    def _getEmpiricalData(self): 
        # load anatomical data
        loaddata    = self.pathData + self.nameDTI
        dti         = sio.loadmat(loaddata)     # C: structural connectivity, D: distance between areas.
        self.D      = dti['D']                  # Distances beween nodes
        anato       = dti['C']                  # Strucural/anatomical network
        self.C      = np.shape(self.D)[1]       # number of nodes/brain areas
        self.eta    = np.float32(1.0 / self.C)             # proportion of oscillator per community
        #self.anato  = anato / np.mean(anato[~np.identity(self.C,dtype=bool)])   # normalize structural network to a mean = 1
        self.anato  = anato / np.mean(anato)
        # load functional data
        loaddata    = self.pathData + self.nameFC
        empiri      = sio.loadmat(loaddata)              # fBands: frequency bands, FCf:functional connectivity
        self.fBands = empiri['freq_bands'].astype(np.float32) # bandpass filter frequency bands
        self.empiFC      = empiri['FC_Env_mean']              # empiprical functional connectivity
        self.empiProfile = []
        for ix in range(0,self.fBands.shape[0]):         # Profile Empirical data
            empi1            = self.empiFC[ix,...]
            self.empiProfile = np.append(self.empiProfile, empi1[np.triu_indices(self.C,1)])
        self.empiProfile = np.clip(self.empiProfile, a_min=0, a_max=None)
        
    def _desingFilterBands(self):
        nyq   = self.fs / 2.0
        trans = 2.0
        self.coeFil = []
        for freq in self.fBands:
            # Filter frequency bands
            passCut = freq / nyq
            stopCut = [(freq[0] - trans) / nyq, (freq[1] + trans) / nyq]
            self.coeFil.append(sg.iirdesign(passCut, stopCut, gpass=0.0025, gstop=30.0,
                                            analog=False, ftype='cheby2', output='sos'))
        # Filter envelops
        self.coeFilEnv = sg.iirdesign(0.5 / nyq, (0.5+trans)/nyq , gpass=0.0025, gstop=30.0,
                                            analog=False, ftype='cheby2', output='sos')
    
    def _doKuramotoOrder(self, z):
        # global order
        ordG        = np.abs( np.mean( z[:,int(self.tMin*self.fs):], axis = 0 ))
        orderG      = np.mean(ordG)
        orderGstd   = np.std(ordG)
        # local order
        ordL        = np.mean( np.abs(z[:,int(self.tMin*self.fs):]), axis = 0 )
        orderL      = np.mean(ordL)
        orderLstd   = np.std(ordL)
        return orderG, orderGstd, orderL, orderLstd      
    
    def fitness(self,x):
        vel     = x[0]
        kG      = x[1]
        kL      = x[2:]     # Kl is an scalar if kL is fix for all nodes, or kL is an array if kL is free
        kS      = self.getAnatoCoupling(kG,kL)
        dlayStep, maxDlay = self.getDelays(vel)
        r, phi  = self._doNodeContainers(maxDlay)
        dlayIdx = self.doIndexDelay(r,dlayStep)
        
        # scales by dt,  try to reduce floating point error, and speed-up
        kS_ = np.float32(kS*self.dt)
        omga_ = np.float32(self.omega*self.dt)
        dlt_ = np.float32( - self.dlt * self.dt)
        
        z    = KAOnodes._KMAOcommu(phi,r,maxDlay,dlayIdx,self.eta,dlt_,self.fs,self.dt,kS_,omga_)    
        #self.z = z
        fit         = self._fitFilterBands(z)
        orderG, orderGstd, orderL, orderLstd = self._doKuramotoOrder(z)
        self.log.append( [fit,vel,orderL,orderG,orderLstd,orderGstd,kG, kL] )
        return np.array([fit])
    
    def doIndexDelay(self,r,dlayStep):
        nodes   = np.tile(np.arange(self.C),(self.C,1)) * r.shape[1]
        outpu   = nodes - dlayStep
        return outpu
    
    def getAnatoCoupling(self,kG,kL):
        """Get anatomical network with couplings"""
        kS = self.anato * kG / self.C        # Globa  coupling
        np.fill_diagonal(kS,kL)              # Local coupling
        return np.float64(kS)
    
    def getDelays(self,vel):
        """Return maximum delay and delay steps in samples"""
        dlay        = self.D / (1000.0 * vel)                     # [seconds] correct from mm to m
        dlayStep    = np.around(dlay / self.dt).astype(np.int32)  # delay on steps backwards to be done
        maxDlay     = np.int32(np.max(dlayStep))                  # number of time steps for the longest delay
        return dlayStep, maxDlay
    
    def _fitFilterBands(self,z):
        ccoef = [None] * self.fBands.shape[0]
        for idx, coefsos in enumerate(self.coeFil):
            # filter frequency bands
            zFilt   = sg.sosfiltfilt(coefsos, np.imag(z), axis=1, padtype='odd')
            zEnv    = np.abs(sg.hilbert(zFilt, axis=1))
            # filter envelope
            zEnvFilt= sg.sosfiltfilt(self.coeFilEnv, zEnv, axis=1, padtype='odd')
            # Correlation discarding warmup time
            envCo   = np.corrcoef(zEnvFilt[:,int(self.tMin*self.fs):-int(self.tMin*self.fs/2)], rowvar=True)
            # set to zero negative correlations
            envCo   = np.clip(envCo, a_min=0, a_max=None)
            simuProfile = envCo[np.triu_indices(z.shape[0],1)]
            ccoef[idx], pval = pearsonr(simuProfile, self.empiProfile[self.edgesBand[0,idx]:self.edgesBand[1,idx]])
        return -1 * np.array(ccoef)
    
    #complex128[:,:](float32[:,:],float32[:,:],int32,int32[:,:],float32,float32,float32,float32,float32[:,:],float32), 
    @jit(nopython=True,cache=True,nogil=True,parallel=True,fastmath=True)
    def _KMAOcommu(phi,r,maxDlay,dlayStep,eta,dlt,fs,dt,kS,omga):
        C       = phi.shape[0]
        pi2     = np.float32(2 * np.pi)
        eta2    = np.float32(0.5 * eta)
        sumRsp  = np.zeros((C),dtype=np.float32)
        sumPHIsp= np.zeros((C),dtype=np.float32)
        for n in range(maxDlay,phi.shape[1]-1):
            rsum1       = dlt * r[:,n]
            rpro1       = eta2 * ( 1 - r[:,n] * r[:,n])
            phipro1     = eta2 * (r[:,n] + 1 / r[:,n])
            idD         = n + dlayStep
            phi_r = phi.ravel()
            r_r   = r.ravel()
            for s in prange(C):
                phiDif      = phi_r[idD[s,:]] - phi[s,n]
                kSr         = kS[:,s] * r_r[idD[s,:]]
                sumRsp[s]   = np.sum( kSr * np.cos( phiDif ))
                sumPHIsp[s] = np.sum( kSr * np.sin( phiDif ))
            rdt         = rsum1 + rpro1 * sumRsp
            phidt       = omga + phipro1 * sumPHIsp
            # add differntial step
            r[:,n+1]    = r[:,n] + rdt
            phi[:,n+1]  = np.remainder(phi[:,n] + phidt, pi2)
        r       = r[:,maxDlay+1:]     # remove history samples used in the begining
        phi     = phi[:,maxDlay+1:]
        # simple downsampling (there may be aliasing)
        r   = r[:,::np.int32(1./(fs*dt))] 
        phi = phi[:,::np.int32(1./(fs*dt))]
        return r * np.exp(1j* phi)
    
    def _doNodeContainers(self,maxDlay):      
        # node's variables
        r           = np.zeros((self.C, int(self.tMax/self.dt + maxDlay)), dtype=np.float32) # node phase parameter [C, Nsamples to integrate]
        phi         = np.zeros((self.C, int(self.tMax/self.dt + maxDlay)), dtype=np.float32) # node phase parameter [C, Nsamples to integrate]
        # initial conditions as history for the time delays
        if self.iniCond == 1:   # random 
            np.random.seed(37)
            phiRa   = np.tile(np.random.uniform(-np.pi,np.pi,self.C), (maxDlay+1,1)).T
            time    = np.tile(np.linspace(0, (maxDlay+1)*self.dt, maxDlay+1, dtype=np.float32),(self.C,1))
            phi[:,0:maxDlay+1]  = np.float32(np.remainder(time * self.omega + phiRa, 2*np.pi))
            np.random.seed(37)
            r[:,0:maxDlay+1]    = np.float32(np.tile(np.random.random(self.C), (maxDlay+1,1))).T
        elif self.iniCond == 2: # phase equaly distributed around the circle 
            omegaT      = self.omega * np.linspace(0, maxDlay*self.dt+self.dt,maxDlay+1, dtype=np.float32)
            r[:,0:maxDlay+1]    = 0.3 * np.ones((self.C,maxDlay+1),dtype=np.float32)
            phi[:,0:maxDlay+1]  = np.tile(np.remainder(omegaT,2*np.pi),(self.C,1))
        return r, phi    

class KAOnodesMultiObjConstr():
    """ KM model with AO reduction. Optimized delay, global, and local coupling.
        Each frequency band is one cost function, to be solved as multi-objective 
        optimization problem. It has two inequality constrains to bound the global order
    """
    def __init__(self):
        self.iniCond  = 1 # Initial conditions - 1: psudo-random, 2: phi uniform, r const(0.3), other: zeros
        self.tMax     = np.float32(4.5)
        self.tMin     = np.float32(1.0)
        self.fs       = np.float32(250.0)
        self.omega    = np.float64(2*np.pi * 40.0)
        self.dlt      = np.float64(1.0)
        self.pathData = '/Users/p277634/python/kaoModel/'
        self.nameDTI  = 'AAL_matrices.mat'         # anatomical network
        self.nameFC   = 'Real_Band_FC.mat'     # Empirical Functional connectivity
        self.dt       = np.float64(5e-4)
        self._getEmpiricalData()
        self._desingFilterBands()
        self.log      = []
        
    def get_mylogs(self):
        return self.log
    
    def get_name(self):
        return "kao with all nodes, multiobjective optimization with constrains"
    
    def get_nobj(self):
        return self.fBands.shape[0]
 
    def get_nic(self):
         """ Number of inequality constrains """
         return 2    
    
    def get_bounds(self):
        """Boundaries on: velocity, kG, kL"""
        upbound     = [3500] * self.C
        upbound     = [20, 9000] + upbound
        lowbound    = [0.1] * self.C
        lowbound    = [1.0, 0.1] + lowbound
        return (lowbound, upbound)
          
    def _getEmpiricalData(self): 
        # load anatomical data
        loaddata    = self.pathData + self.nameDTI
        dti         = sio.loadmat(loaddata)     # C: structural connectivity, D: distance between areas.
        self.D      = dti['D']                  # Distances beween nodes
        anato       = dti['C']                  # Strucural/anatomical network
        self.C      = np.shape(self.D)[1]       # number of nodes/brain areas
        self.eta    = np.float32(1.0 / self.C)             # proportion of oscillator per community
        #self.anato  = anato / np.mean(anato[~np.identity(self.C,dtype=bool)])   # normalize structural network to a mean = 1
        self.anato  = anato / np.mean(anato)
        # load functional data
        loaddata    = self.pathData + self.nameFC
        empiri      = sio.loadmat(loaddata)              # fBands: frequency bands, FCf:functional connectivity
        self.fBands = empiri['freq_bands'].astype(np.float32) # bandpass filter frequency bands
        self.empiFC      = empiri['FC_Env_mean']              # empiprical functional connectivity
        self.empiProfile = []
        for ix in range(0,self.fBands.shape[0]):         # Profile Empirical data
            empi1            = self.empiFC[ix,...]
            self.empiProfile = np.append(self.empiProfile, empi1[np.triu_indices(self.C,1)])
        self.empiProfile = np.clip(self.empiProfile, a_min=0, a_max=None)
        
    def _desingFilterBands(self):
        nyq   = self.fs / 2.0
        trans = 2.0
        self.coeFil = []
        for freq in self.fBands:
            # Filter frequency bands
            passCut = freq / nyq
            stopCut = [(freq[0] - trans) / nyq, (freq[1] + trans) / nyq]
            self.coeFil.append(sg.iirdesign(passCut, stopCut, gpass=0.0025, gstop=30.0,
                                            analog=False, ftype='cheby2', output='sos'))
        # Filter envelops
        self.coeFilEnv = sg.iirdesign(0.5 / nyq, (0.5+trans)/nyq , gpass=0.0025, gstop=30.0,
                                            analog=False, ftype='cheby2', output='sos')
    
    def _doKuramotoOrder(self, z):
        # global order
        order       = np.mean( z[:,int(self.tMin*self.fs):], axis = 0 )
        ordG        = np.abs( order )
        orderG      = np.mean(ordG)
        orderGstd   = np.std(ordG)
        # local order
        ordL        = np.mean( np.abs(z[:,int(self.tMin*self.fs):]), axis = 0 )
        orderL      = np.mean(ordL)
        orderLstd   = np.std(ordL)
        # global order as if local ordedr is one
        order1 = np.mean( np.angle( order ))
        
        return orderG, orderGstd, orderL, orderLstd, order1    
    
    def fitness(self,x):
        vel     = x[0]
        kG      = x[1]
        kL      = x[2:]     # Kl is an scalar if kL is fix for all nodes, or kL is an array if kL is free
        kS      = self.getAnatoCoupling(kG,kL)
        dlayStep, maxDlay = self.getDelays(vel)
        r, phi  = self._doNodeContainers(maxDlay)
        dlayIdx = self.doIndexDelay(r,dlayStep)
        
        # scales by dt,  try to reduce floating point error, and speed-up
        kS_ = np.float32(kS*self.dt)
        omga_ = np.float32(self.omega*self.dt)
        dlt_ = np.float32( - self.dlt * self.dt)
        
        z    = KAOnodes._KMAOcommu(phi,r,maxDlay,dlayIdx,self.eta,dlt_,self.fs,self.dt,kS_,omga_)    
        #self.z = z
        fit         = self._fitFilterBands(z)
        orderG, orderGstd, orderL, orderLstd, order1  = self._doKuramotoOrder(z)
        self.log.append(np.append( [fit,vel,orderL,orderG,orderLstd,orderGstd,kG] , kL))
        return np.append(fit, np.array([0.1 - order1, order1 - 0.6]))
    
    def doIndexDelay(self,r,dlayStep):
        nodes   = np.tile(np.arange(self.C),(self.C,1)) * r.shape[1]
        outpu   = nodes - dlayStep
        return outpu
    
    def getAnatoCoupling(self,kG,kL):
        """Get anatomical network with couplings"""
        kS = self.anato * kG / self.C        # Globa  coupling
        np.fill_diagonal(kS,kL)              # Local coupling
        return np.float64(kS)
    
    def getDelays(self,vel):
        """Return maximum delay and delay steps in samples"""
        dlay        = self.D / (1000.0 * vel)                     # [seconds] correct from mm to m
        dlayStep    = np.around(dlay / self.dt).astype(np.int32)  # delay on steps backwards to be done
        maxDlay     = np.int32(np.max(dlayStep))                  # number of time steps for the longest delay
        return dlayStep, maxDlay
    
    def _fitFilterBands(self,z):
        ccoef = [None] * self.fBands.shape[0]
        for idx, coefsos in enumerate(self.coeFil):
            # filter frequency bands
            zFilt   = sg.sosfiltfilt(coefsos, np.imag(z), axis=1, padtype='odd')
            zEnv    = np.abs(sg.hilbert(zFilt, axis=1))
            # filter envelope
            zEnvFilt= sg.sosfiltfilt(self.coeFilEnv, zEnv, axis=1, padtype='odd')
            # Correlation discarding warmup time
            envCo   = np.corrcoef(zEnvFilt[:,int(self.tMin*self.fs):-int(self.tMin*self.fs/2)], rowvar=True)
            # set to zero negative correlations
            envCo   = np.clip(envCo, a_min=0, a_max=None)
            simuProfile = envCo[np.triu_indices(z.shape[0],1)]
            ccoef[idx], pval = pearsonr(simuProfile, self.empiProfile[self.edgesBand[0,idx]:self.edgesBand[1,idx]])
        return -1 * np.array(ccoef)
    
    #complex128[:,:](float32[:,:],float32[:,:],int32,int32[:,:],float32,float32,float32,float32,float32[:,:],float32), 
    @jit(nopython=True,cache=True,nogil=True,parallel=True,fastmath=True)
    def _KMAOcommu(phi,r,maxDlay,dlayStep,eta,dlt,fs,dt,kS,omga):
        C       = phi.shape[0]
        pi2     = np.float32(2 * np.pi)
        eta2    = np.float32(0.5 * eta)
        sumRsp  = np.zeros((C),dtype=np.float32)
        sumPHIsp= np.zeros((C),dtype=np.float32)
        for n in range(maxDlay,phi.shape[1]-1):
            rsum1       = dlt * r[:,n]
            rpro1       = eta2 * ( 1 - r[:,n] * r[:,n])
            phipro1     = eta2 * (r[:,n] + 1 / r[:,n])
            idD         = n + dlayStep
            phi_r = phi.ravel()
            r_r   = r.ravel()
            for s in prange(C):
                phiDif      = phi_r[idD[s,:]] - phi[s,n]
                kSr         = kS[:,s] * r_r[idD[s,:]]
                sumRsp[s]   = np.sum( kSr * np.cos( phiDif ))
                sumPHIsp[s] = np.sum( kSr * np.sin( phiDif ))
            rdt         = rsum1 + rpro1 * sumRsp
            phidt       = omga + phipro1 * sumPHIsp
            # add differntial step
            r[:,n+1]    = r[:,n] + rdt
            phi[:,n+1]  = np.remainder(phi[:,n] + phidt, pi2)
        r       = r[:,maxDlay+1:]     # remove history samples used in the begining
        phi     = phi[:,maxDlay+1:]
        # simple downsampling (there may be aliasing)
        r   = r[:,::np.int32(1./(fs*dt))] 
        phi = phi[:,::np.int32(1./(fs*dt))]
        return r * np.exp(1j* phi)
    
    def _doNodeContainers(self,maxDlay):      
        # node's variables
        r           = np.zeros((self.C, int(self.tMax/self.dt + maxDlay)), dtype=np.float32) # node phase parameter [C, Nsamples to integrate]
        phi         = np.zeros((self.C, int(self.tMax/self.dt + maxDlay)), dtype=np.float32) # node phase parameter [C, Nsamples to integrate]
        # initial conditions as history for the time delays
        if self.iniCond == 1:   # random 
            np.random.seed(37)
            phiRa   = np.tile(np.random.uniform(-np.pi,np.pi,self.C), (maxDlay+1,1)).T
            time    = np.tile(np.linspace(0, (maxDlay+1)*self.dt, maxDlay+1, dtype=np.float32),(self.C,1))
            phi[:,0:maxDlay+1]  = np.float32(np.remainder(time * self.omega + phiRa, 2*np.pi))
            np.random.seed(37)
            r[:,0:maxDlay+1]    = np.float32(np.tile(np.random.random(self.C), (maxDlay+1,1))).T
        elif self.iniCond == 2: # phase equaly distributed around the circle 
            omegaT      = self.omega * np.linspace(0, maxDlay*self.dt+self.dt,maxDlay+1, dtype=np.float32)
            r[:,0:maxDlay+1]    = 0.3 * np.ones((self.C,maxDlay+1),dtype=np.float32)
            phi[:,0:maxDlay+1]  = np.tile(np.remainder(omegaT,2*np.pi),(self.C,1))
        return r, phi    
    

class KAOnodes_noVel():
    """ Kuramoto Antonsen-Ott model. Free parameters to be optimized are Global 
        coupling and local cupling
    """
    def __init__(self):
        self.iniCond  = 1 # Initial conditions - 1: psudo-random, 2: phi uniform, r const(0.3), other: zeros
        self.tMax     = np.float32(3.5)
        self.tMin     = np.float32(1.0)
        self.fs       = np.float32(250.0)
        self.omega    = np.float64(2*np.pi * 40.0)
        self.dlt      = np.float64(1.0)
        self.pathData = '/home/oscar/Documents/MATLAB/Kuramoto/Cabral/'
        self.nameDTI  = 'AAL_matrices.mat'         # anatomical network
        self.nameFC   = 'Real_Band_FC.mat'     # Empirical Functional connectivity
        self.dt       = np.float64(5e-4)
        self._getEmpiricalData()
        self._desingFilterBands()
        self.log      = np.empty((0, 7 + self.C))
        self.velocity = np.float32(1.25)      # Conduction velocity [m/s]
        self._doDelays()
        self._doNodeContainers()
        self._doIndexDelay()
        
        
    def get_mylogs(self):
        return self.log
    
    def get_name(self):
        return "kaoDynamicsFixKl"
    
    def get_bounds(self):
        """Boundaries on: velocity, kG, kL"""
        upbound     = [1000] * self.C
        upbound     = [5000] + upbound
        lowbound    = [1] * self.C
        lowbound    = [1] + lowbound
        return (lowbound, upbound)
          
    def _getEmpiricalData(self): 
        # load anatomical data
        loaddata    = self.pathData + self.nameDTI
        dti         = sio.loadmat(loaddata)     # C: structural connectivity, D: distance between areas.
        self.D      = dti['D']                  # Distances beween nodes
        anato       = dti['C']                  # Strucural/anatomical network
        self.C      = np.shape(self.D)[1]       # number of nodes/brain areas
        self.eta    = np.float32(1.0 / self.C)             # proportion of oscillator per community
        #self.anato  = anato / np.mean(anato[~np.identity(self.C,dtype=bool)])   # normalize structural network to a mean = 1
        self.anato  = anato / np.mean(anato) 
        # load functional data
        loaddata    = self.pathData + self.nameFC
        empiri      = sio.loadmat(loaddata)              # fBands: frequency bands, FCf:functional connectivity
        self.fBands = empiri['freq_bands'].astype(float) # bandpass filter frequency bands
        self.empiFC      = empiri['FC_Env_mean']              # empiprical functional connectivity
        self.empiProfile = []
        for ix in range(0,self.fBands.shape[0]):         # Profile Empirical data
            empi1            = self.empiFC[ix,...]
            self.empiProfile = np.append(self.empiProfile, empi1[np.triu_indices(self.C,1)])
        self.empiProfile = np.clip(self.empiProfile, a_min=0, a_max=None)
        
    def _desingFilterBands(self):
        nyq   = self.fs / 2.0
        trans = 2.0
        self.coeFil = []
        for freq in self.fBands:
            # Filter frequency bands
            passCut = freq / nyq
            stopCut = [(freq[0] - trans) / nyq, (freq[1] + trans) / nyq]
            self.coeFil.append(sg.iirdesign(passCut, stopCut, gpass=0.0025, gstop=30.0,
                                            analog=False, ftype='cheby2', output='sos'))
        # Filter envelops
        self.coeFilEnv = sg.iirdesign(0.5 / nyq, (0.5+trans)/nyq , gpass=0.0025, gstop=30.0,
                                            analog=False, ftype='cheby2', output='sos')
        
    def _doDelays(self):
        """Return maximum delay and delay steps in samples"""
        dlay            = self.D / (1000.0 * self.velocity)            # [seconds] correct from mm to m
        self.dlayStep   = np.around(dlay / self.dt).astype(np.int32)  # delay on steps backwards to be done
        self.maxDlay    = np.int32(np.max(self.dlayStep))                  # number of time steps for the longest delay

    def _doIndexDelay(self):
        nodes           = np.tile(np.arange(self.C),(self.C,1)) * self.r.shape[1]
        self.dlayIdx    = nodes - self.dlayStep
    
    def _doNodeContainers(self,maxDlay):      
        # node's variables
        r           = np.zeros((self.C, int(self.tMax/self.dt + maxDlay)), dtype=np.float32) # node phase parameter [C, Nsamples to integrate]
        phi         = np.zeros((self.C, int(self.tMax/self.dt + maxDlay)), dtype=np.float32) # node phase parameter [C, Nsamples to integrate]
        # initial conditions as history for the time delays
        if self.iniCond == 1:   # random 
            np.random.seed(37)
            phiRa   = np.tile(np.random.uniform(-np.pi,np.pi,self.C), (maxDlay+1,1)).T
            time    = np.tile(np.linspace(0, (maxDlay+1)*self.dt, maxDlay+1, dtype=np.float32),(self.C,1))
            phi[:,0:maxDlay+1]  = np.float32(np.remainder(time * self.omega + phiRa, 2*np.pi))
            np.random.seed(37)
            r[:,0:maxDlay+1]    = np.float32(np.tile(np.random.random(self.C), (maxDlay+1,1))).T
        elif self.iniCond == 2: # phase equaly distributed around the circle 
            omegaT      = self.omega * np.linspace(0, maxDlay*self.dt+self.dt,maxDlay+1, dtype=np.float32)
            r[:,0:maxDlay+1]    = 0.3 * np.ones((self.C,maxDlay+1),dtype=np.float32)
            phi[:,0:maxDlay+1]  = np.tile(np.remainder(omegaT,2*np.pi),(self.C,1))
        return r, phi
    
    def _doKuramotoOrder(self, z):
        # global order
        ordG        = np.abs( np.mean( z[:,int(self.tMin*self.fs):], axis = 0 ))
        orderG      = np.mean(ordG)
        orderGstd   = np.std(ordG)
        # local order
        ordL        = np.mean( np.abs(z[:,int(self.tMin*self.fs):]), axis = 0 )
        orderL      = np.mean(ordL)
        orderLstd   = np.std(ordL)        
        return orderG, orderGstd, orderL, orderLstd    
    
    def fitness(self,x):
        kG      = x[0]
        kL      = x[1:]     # Kl is an scalar if kL is fix for all nodes, or kL is an array if kL is free
        kS      = self.getAnatoCoupling(kG,kL)
        
       # scales by dt, try to reduce floating point error, and speed-up
        kS_ = np.float32(kS*self.dt)
        omga_ = np.float32(self.omega*self.dt)
        dlt_ = np.float32(-self.dlt * self.dt)
        
        z    = KAOnodes_noVel._KMAOcommu(self.phi,self.r,self.maxDlay,self.dlayIdx,self.eta,dlt_,self.fs,self.dt,kS_,omga_)
        #self.z = z
        fit         = self._fitFilterBands(z)
        orderG, orderGsd, orderL, orderLsd = self._doKuramotoOrder(z)
        self.log    = np.vstack((self.log,
                                 np.append( [fit,self.velocity,orderL,orderG,orderLsd,orderGsd,kG] , kL)))
        return np.array([fit])
    
    
    def getAnatoCoupling(self,kG,kL):
        """Get anatomical network with couplings"""
        kS = self.anato * kG / self.C        # Globa  coupling
        np.fill_diagonal(kS,kL)              # Local coupling
        return np.float64(kS)
    
    
    def _fitFilterBands(self,z):
        simuProfile = []
        for coefsos in self.coeFil:
            # filter frequency bands
            zFilt   = sg.sosfiltfilt(coefsos, np.imag(z), axis=1, padtype='odd')
            zEnv    = np.abs(sg.hilbert(zFilt, axis=1))
            # filter envelope
            zEnvFilt= sg.sosfiltfilt(self.coeFilEnv, zEnv, axis=1, padtype='odd')
            # Correlation discarding warmup time
            envCo = np.corrcoef(zEnvFilt[:,int(self.tMin*self.fs):-int(self.tMin*self.fs/2)], rowvar=True)
            # set to zero negative correlations
            envCo = np.clip(envCo, a_min=0, a_max=None)
            simuProfile  = np.append(simuProfile, envCo[np.triu_indices(z.shape[0],1)])
        ccoef, pval = pearsonr(simuProfile, self.empiProfile)
        return -1 * ccoef
    
    #complex128[:,:](float32[:,:],float32[:,:],int32,int32[:,:],float32,float32,float32,float32,float32[:,:],float32), 
    @jit(nopython=True,cache=True,nogil=True,parallel=True,fastmath=True)
    def _KMAOcommu(phi,r,maxDlay,dlayStep,eta,dlt,fs,dt,kS,omga):
        C       = phi.shape[0]
        pi2     = np.float32(2 * np.pi)
        eta2    = np.float32(0.5 * eta)
        sumRsp  = np.empty((C),dtype=np.float32)
        sumPHIsp= np.empty((C),dtype=np.float32)
        for n in range(maxDlay,phi.shape[1]-1):
            rsum1       = -dlt * r[:,n]
            rpro1       = eta2 * ( 1 - r[:,n] * r[:,n])
            phipro1     = eta2 * (r[:,n] + 1 / r[:,n])
            idD   = n - dlayStep
            phi_r = phi.ravel()
            r_r   = r.ravel()
            for s in prange(C):
                phiDif      = phi_r[idD[:,s]] - phi[s,n]
                kSr         = kS[:,s] * r_r[idD[:,s]]
                sumRsp[s]   = np.sum( kSr * np.cos( phiDif ))
                sumPHIsp[s] = np.sum( kSr * np.sin( phiDif ))
            rdt         = rsum1 + rpro1 * sumRsp
            phidt       = omga + phipro1 * sumPHIsp
            # add differntial step
            r[:,n+1]    = r[:,n] + rdt
            phi[:,n+1]  = np.remainder(phi[:,n] + phidt, pi2)
        r       = r[:,maxDlay+1:]     # remove history samples used in the begining
        phi     = phi[:,maxDlay+1:]
        # simple downsampling (there may be aliasing)
        r   = r[:,::np.int32(1./(fs*dt))] 
        phi = phi[:,::np.int32(1./(fs*dt))]
        return r * np.exp(1j* phi)
    
