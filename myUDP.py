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

#from numba import float32, int64, complex128, prange

class Testkao():
    def __init__(self):
        self.tMax     = np.float32(3.5)
        self.tMin     = np.float32(1.0)
        self.fs       = np.float32(250.0)
        self.omega    = np.float32(2*np.pi * 40.0)
        self.dlt      = np.float32(1.0)
        self.pathData = '/home/oscar/Documents/MATLAB/Kuramoto/Cabral/'
        self.nameDTI  = 'AAL_matrices.mat'         # anatomical network
        self.nameFC   = 'Real_Band_FC.mat'     # Empirical Functional connectivity
        self.dt       = np.float32(2e-4)
        self._getEmpiricalData()
        self._desingFilterBands()
        self.log      = np.empty((0,8))
        
    def get_mylogs(self):
        return self.log
    
    def get_name(self):
        return "kaoDynamicsFixKl"
          
    def _getEmpiricalData(self): 
        # load anatomical data
        loaddata    = self.pathData + self.nameDTI
        dti         = sio.loadmat(loaddata)     # C: structural connectivity, D: distance between areas.
        self.D      = dti['D']                  # Distances beween nodes
        anato       = dti['C']                  # Strucural/anatomical network
        self.C      = np.shape(self.D)[1]       # number of nodes/brain areas
        self.eta    = np.float32(1.0 / self.C)             # proportion of oscillator per community
        self.anato  = anato / np.mean(anato[~np.identity(self.C,dtype=bool)])   # normalize structural network to a mean = 1
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
        kL      = x[1]
        kG      = x[2]
        kS      = self.getAnatoCoupling(kG,kL)
        dlayStep, maxDlay = self.getDelays(vel)
        r, phi  = self._doNodeContainers(maxDlay)
        dlayIdx = self.doIndexDelay(r,dlayStep)
        z    = Testkao._KMAOcommu(phi,r,maxDlay,dlayIdx,self.eta,self.dlt,self.fs,self.dt,kS,self.omega)
        #self.z = z
        fit, self.simuProfile = self._fitFilterBands(z)
        orderG, orderGstd, orderL, orderLstd = self._doKuramotoOrder(z)
        self.log= np.vstack((self.log, np.array([fit,vel,kL,kG,orderG,orderGstd,orderL,orderLstd])))
        return np.array([fit])
    
    def doIndexDelay(self,r,dlayStep):
        nodes   = np.tile(np.arange(self.C),(self.C,1)) * r.shape[1]
        outpu   = nodes - dlayStep
        return outpu
    
    def getAnatoCoupling(self,kG,kL):
        """Get anatomical network with couplings"""
        kS = self.anato * kG / self.C        # Globa  coupling
        np.fill_diagonal(kS,kL)              # Local coupling
        return kS
    
    def getDelays(self,vel):
        """Return maximum delay and delay steps in samples"""
        dlay        = self.D / (1000.0 * vel)                # [seconds] correct from mm to m
        dlayStep    = np.around(dlay / self.dt).astype(np.int64)  # delay on steps backwards to be done
        maxDlay     = np.int64(np.max(dlayStep))                  # number of time steps for the longest delay
        return dlayStep, maxDlay
    
    def get_bounds(self):
        """Boundaries on: velocity, Kl, Kg"""
        return ([0.1, 1, 1],[25, 1000, 5000])
    
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
        return -1 * ccoef, simuProfile
    
    #complex128[:,:](float32[:,:],float32[:,:],int64,int64[:,:],float32,float32,float32,float32,float32[:,:],float32), 
    @jit(nopython=True,cache=True,nogil=True,parallel=True,fastmath=False)
    def _KMAOcommu(phi,r,maxDlay,dlayStep,eta,dlt,fs,dt,kS,omga):
        C       = phi.shape[0]
        pi2     = np.float32(2 * np.pi)
        eta2    = np.float32(0.5 * eta)
        sumRsp  = np.empty((C),dtype=np.float32)
        sumPHIsp= np.empty((C),dtype=np.float32)
        for n in range(maxDlay,phi.shape[1]-1):
            rsum1       = -dlt * r[:,n]
            rpro1       = eta2 * ( 1 - r[:,n] * r[:,n])# r*r
            phipro1     = eta2 * (r[:,n] * r[:,n] + 1) / r[:,n]
            idD = n + dlayStep
            #for s in nodes:
            for s in prange(C):
                #idD         = n - dlayStep[:,s] + commuOff
                phiDif      = phi.ravel()[idD[s,:]] - phi[s,n]
                kSr         = kS[:,s] * r.ravel()[idD[s,:]]
                sumRsp[s]   = np.sum( kSr * np.cos( phiDif ))
                sumPHIsp[s] = np.sum( kSr * np.sin( phiDif ))
            rdt         = rsum1 + rpro1 * sumRsp
            phidt       = omga + phipro1 * sumPHIsp
            # add differntial step
            r[:,n+1]    = r[:,n] + dt*rdt
            phi[:,n+1]  = np.remainder(phi[:,n] + dt*phidt, pi2)
        r       = r[:,maxDlay+1:]     # remove history samples used in the begining
        phi     = phi[:,maxDlay+1:]
        # simple downsampling (there may be aliasing)
        r   = r[:,::np.int32(1./(fs*dt))] 
        phi = phi[:,::np.int32(1./(fs*dt))]
        return r * np.exp(1j* phi)
    
    def _doNodeContainers(self,maxDlay):      
        # node's variables
        r           = np.empty((self.C, int(self.tMax/self.dt + maxDlay)), dtype=np.float32) # node phase parameter [C, Nsamples to integrate]
        phi         = np.empty((self.C, int(self.tMax/self.dt + maxDlay)), dtype=np.float32) # node phase parameter [C, Nsamples to integrate]
        # initial conditions as history for the time delays
        omegaT      = self.omega * np.linspace(0, maxDlay*self.dt+self.dt,maxDlay+1, dtype=np.float32)
        r[:,0:maxDlay+1]    = 0.3 * np.ones((self.C,maxDlay+1))
        phi[:,0:maxDlay+1]  = np.tile(np.remainder(omegaT,2*np.pi),(self.C,1))
        return r, phi
    
    def setVariablesModel(self,x):
        "set the basic variable need to run the KM model"
        vel     = x[0]
        kL      = x[1]
        kG      = x[2]
        self.kS      = self.getAnatoCoupling(kG,kL)
        self.dlayStep, self.maxDlay = self.getDelays(vel)
        self.r, self.phi   = self._doNodeContainers(self.maxDlay)
        self.dlayId  = self.doIndexDelay(self.phi,self.dlayStep)
    
    
class kMcabral():
    """ Kuramoto Model by Cabral without noise"""
    def __init__(self):
        self.tMax     = np.float32(3.0)
        self.tMin     = np.float32(1.0)
        self.fs       = np.float32(250.0)
        self.omega    = np.float32(2*np.pi* 40.0)
        self.dlt      = np.float32(1.0)
        self.pathData = '/home/oscar/Documents/MATLAB/Kuramoto/Cabral/'
        self.nameDTI  = 'AAL_matrices.mat'         # anatomical network
        self.nameFC   = 'Real_Band_FC.mat'     # Empirical Functional connectivity
        self.dt       = np.float32(1e-4)
        self._getEmpiricalData()
        self._desingFilterBands()
        self.log      = np.empty((0,5))
        
    def get_mylogs(self):
        return self.log
    
    def get_name(self):
        return "KM Cabral"
          
    def _getEmpiricalData(self): 
        # load anatomical data
        loaddata    = self.pathData + self.nameDTI
        dti         = sio.loadmat(loaddata)     # C: structural connectivity, D: distance between areas.
        self.D      = dti['D']                  # Distances beween nodes
        anato       = dti['C']                  # Strucural/anatomical network
        self.C      = np.shape(self.D)[1]       # number of nodes/brain areas
        self.anato  = anato / np.mean(anato[~np.identity(self.C,dtype=bool)])   # normalize structural network to a mean = 1
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
    
    def _doKuramotoOrder(self, z):
        # global order
        ordG        = np.abs( np.mean( z[:,int(self.tMin*self.fs):], axis = 0 ))
        orderG      = np.mean(ordG)
        orderGstd   = np.std(ordG)
        return orderG, orderGstd,    
    
    def fitness(self,x):
        vel     = x[0]
        kG      = x[1]
        kS      = self.getAnatoCoupling(kG)
        dlayStep, maxDlay = self.getDelays(vel)
        phi     = self._doNodeContainers(maxDlay)
        #dlayIdx = self.doIndexDelay(phi,dlayStep)
        z    = kMcabral._KMcabral(phi,dlayStep,dlayStep,self.fs,self.dt,kS,self.omega)
        #self.z = z
        fit, self.simuProfile = self._fitFilterBands(z)
        orderG, orderGstd = self._doKuramotoOrder(z)
        self.log    = np.vstack(self.log, np.array([fit,vel,kG,orderG,orderGstd]))
        return np.array([fit])
    
    def doIndexDelay(self,r,dlayStep):
        commuOff = np.arange(0,r.shape[0]) * r.shape[1]
        commuOff = np.tile(commuOff,(r.shape[0],1)).T
        outpu = dlayStep + commuOff
        return outpu
    
    def getAnatoCoupling(self,kG):
        """Get anatomical network with couplings"""
        kS = self.anato * kG / self.C        # Globa  coupling
        np.fill_diagonal(kS, 0)              # Local coupling
        return kS
    
    def getDelays(self,vel):
        """Return maximum delay and delay steps in samples"""
        dlay        = self.D / (1000.0 * vel)                # [seconds] correct from mm to m
        dlayStep    = np.around(dlay / self.dt).astype(np.int64)  # delay on steps backwards to be done
        maxDlay     = np.int64(np.max(dlayStep))                  # number of time steps for the longest delay
        return dlayStep, maxDlay
    
    def get_bounds(self):
        """Boundaries on: velocity, Kl, Kg"""
        return ([0.1, 0.01],[25, 5000])
    
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
        return -1 * ccoef, simuProfile
    
    #complex128[:,:](float32[:,:],float32[:,:],int64,int64[:,:],float32,float32,float32,float32,float32[:,:],float32), 
    @jit(nopython=True,cache=True,nogil=True,parallel=True,fastmath=False)
    def _KMcabral(phi,maxDlay,dlayStep,fs,dt,kS,omga):
        C       = phi.shape[0]
        #nodes   = range(0,C)        
        pi2     = np.float32(2 * np.pi)
        phidt   = np.zeros(C, dtype=np.float32)
        for n in range(maxDlay, phi.shape[1]-1):
            idD     = n - dlayStep
            for s in prange(C):
                phiDif      = phi[:, idD[:,s]] - phi[s, n] 
                phidt[s]    = np.sum(kS[:,s] * np.sin(phiDif))
            
            phi[:,n+1]  = np.remainder(phi[:,n] + dt * (phidt+omga), pi2)
            
        phi     = phi[:,maxDlay+1:]
        # simple downsampling (there may be aliasing)
        phi = phi[:,::np.int64(1./(fs*dt))]
        return 1.0 * np.exp(1j* phi)
    
    def _doNodeContainers(self,maxDlay):      
        # node's variables
        #import pdb; pdb.set_trace()
        phi         = np.empty((self.C, int(self.tMax/self.dt + maxDlay)), dtype=np.float32) # node phase parameter [C, Nsamples to integrate]
        # initial conditions as history for the time delays
        omegaT      = self.omega * np.linspace(0, maxDlay*self.dt+self.dt,maxDlay+1, dtype=np.float32)
        phi[:,0:maxDlay+1]  = np.tile(np.remainder(omegaT,2*np.pi),(self.C,1))
        return phi    