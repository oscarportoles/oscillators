#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Set of classes that defines all the approaches to optiizing the parameters in a
Kuramoto Anton-Ott model (KAOM)

Created on Fri Aug 17 11:42:49 2018

@author: oscar
"""
import numpy as np
import scipy.io as sio
from numba import jit, prange
import scipy.signal as sg
from scipy.stats import pearsonr

class KAOmother():
    """ Parent class. Defines methods common to all problems """
    def __init__(self, lib=1):
        self.lib      = lib                 # library to do optimzattion - 1:pagmo, 2:platypus 
        self.lowBound = [1, 0.1, 0.1]       # upper boundary candidates: [velocity, golbal coupling (kG), local coupling (kL)] 
        self.upBound  = [20, 9000, 3000]    # lower boundary candidates: [velocity, golbal coupling (kG), local coupling (kL)] 
        self.tMax     = np.float32(4.5)
        self.tMin     = np.float32(1.0)
        self.fs       = np.float32(250.0)
        self.omega    = np.float64(2*np.pi * 40.0)
        self.dlt      = np.float64(5.0)
        self.pathData = '/home/oscar/Documents/PythonProjects/kuramotoAO/basicsKAO/'
        #self.pathData = '/Users/p277634/python/kaoModel/'
        self.nameDTI  = 'AAL_matrices.mat'         # anatomical network
        self.nameFC   = 'Real_Band_FC.mat'     # Empirical Functional connectivity
        self.dt       = np.float64(5e-4)
        self.iniCond  = 1               # Initial conditions - 1: psudo-random, 2: phi uniform, r constant(0.3), other: zeros
        self._getEmpiricalData()
        self._desingFilterBands()
        self.log      = []
                
    def get_mylogs(self):
        return self.log
    
    def get_name(self):
        return "Mother class for al problems"
    
    def get_bounds(self):
        """ Boundaries on: velocity, golbal coupling (kG), local coupling (kL).
            Global order is almost one with velocity hgher than 20
        """
        return (self.lowBound,self.upBound)
    
    def _getEmpiricalData(self): 
        # load anatomical data
        loaddata    = self.pathData + self.nameDTI
        dti         = sio.loadmat(loaddata)     # C: structural connectivity, D: distance between areas.
        self.D      = dti['D']                  # Distances beween nodes
        anato       = dti['C']                  # Strucural/anatomical network
        self.C      = np.shape(self.D)[1]       # number of nodes/brain areas
        self.eta    = np.float32(1.0 / self.C)             # proportion of oscillator per community
        #self.anato  = anato / np.mean(anato[~np.identity(self.C,dtype=bool)])   # normalize structural network to a mean = 1
        self.anato  = anato / np.mean(anato)   # normalize structural network to a mean = 1

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
        # get edges of each frequency band to cut self.empiProfile
        nBands          = self.fBands.shape[0]
        lenBand         = self.empiProfile.shape[0] / self.fBands.shape[0]
        self.edgesBand  = np.array([np.arange(0,nBands)*lenBand, np.arange(1,nBands+1)*lenBand], dtype=np.int32)

        
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
            simuProfile.append(envCo[np.triu_indices(z.shape[0],1)])
        #print(simuProfile.shape)
        #ccoef, pval = pearsonr(simuProfile, self.empiProfile)
        return simuProfile
        
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
        dlay        = self.D / (1000.0 * vel)                # [seconds] correct from mm to m
        dlayStep    = np.around(dlay / self.dt).astype(np.int32)  # delay on steps backwards to be done
        maxDlay     = np.int32(np.max(dlayStep))                  # number of time steps for the longest delay
        return dlayStep, maxDlay
    
    def _doKuramotoOrder(self, z):
        # global ordedr with local order == 1
        order       = 1 * np.exp(1j * np.angle(z[:,int(self.tMin*self.fs):]))  
        order       = np.mean(np.abs( np.mean( order, axis = 0 )))
        # global order
        ordG        = np.abs( np.mean( z[:,int(self.tMin*self.fs):], axis = 0 ))
        orderG      = np.mean(ordG)
        orderGstd   = np.std(ordG)
        # local order
        ordL        = np.mean( np.abs(z[:,int(self.tMin*self.fs):]), axis = 0 )
        orderL      = np.mean(ordL)
        orderLstd   = np.std(ordL)
        return orderG, orderGstd, orderL, orderLstd, order
    
    #complex128[:,:](float32[:,:],float32[:,:],int64,int64[:,:],float32,float32,float32,float32,float32[:,:],float32), 
    @jit(nopython=True,cache=True,nogil=True,parallel=True,fastmath=True)
    def _KMAOcommu(phi,r,maxDlay,dlayStep,eta,dlt,fs,dt,kS,omga):
        C       = phi.shape[0]
        pi2     = np.float32(2.0 * np.pi)
        eta2    = np.float32(0.5 * eta)
        sumRsp  = np.empty((C), dtype=np.float32)
        sumPHIsp= np.empty((C), dtype=np.float32)
        for n in range(maxDlay,phi.shape[1]-1):
            rsum1       = dlt * r[:,n]
            rpro1       = eta2 * ( 1 - r[:,n] * r[:,n])# r*r
            phipro1     = eta2 * (r[:,n] + 1 / r[:,n])
            idD = n + dlayStep
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
    
    def setVariablesModel(self,x):
        """set the basic variable need to run the KM model 
        [velovity, global coupling, local coupling] """
        vel     = x[0]
        kG      = x[1]
        kL      = x[2:]
        self.kS      = self.getAnatoCoupling(kG,kL)
        self.dlayStep, self.maxDlay = self.getDelays(vel)
        self.r, self.phi   = self._doNodeContainers(self.maxDlay)
        self.dlayIdx  = self.doIndexDelay(self.phi,self.dlayStep)
        
    def doDiffConnecti_SO(self, profil):
        """ Single Objective fucntion. Difrence of empirical and simulated connectivity.
            profil is a list with a numpy arry for each frequency band """
        ccoef, pval = pearsonr(np.array(profil).ravel(), self.empiProfile)
        fit = np.sum( (np.array(profil).ravel() - self.empiProfile) ** 2 )    
        return fit, ccoef
  
    def doDiffConnecti_MO(self, profil):
        """ Multi Obejctive function. Difrence of empirical and simulated connectivity. 
            profil is a list  with a numpy arry for each frequency band """
        ccoef = [None] * self.fBands.shape[0]
        fit   = np.zeros(self.fBands.shape[0])
        for idx, profiBand in enumerate(profil):
            ccoef[idx], pval = pearsonr(profiBand, self.empiProfile[self.edgesBand[0,idx]:self.edgesBand[1,idx]])
            fit[idx] =  np.sum( (profiBand - self.empiProfile[self.edgesBand[0,idx]:self.edgesBand[1,idx]]) ** 2 )
        return fit, np.array( ccoef )
      
######### ------------------------------------------------ ##############
#               Definition of individual problems
######### ------------------------------------------------ ##############
    
    
class KAOsimple(KAOmother):
    """ Local coupling is the same for all nodes """
    def get_name(self):
        return "KAOM with fix kL for all nodesl"
    
    def fitness(self,x):
        vel     = x[0]
        kG      = x[1]
        kL      = x[2]
        kS      = self.getAnatoCoupling(kG,kL)
        dlayStep, maxDlay = self.getDelays(vel)
        r, phi  = self._doNodeContainers(maxDlay)
        dlayIdx = self.doIndexDelay(r,dlayStep)
        
        # scales by dt, try to reduce floating point error, and speed-up
        kS_ = np.float32(kS*self.dt)
        omga_ = np.float32(self.omega*self.dt)
        dlt_ = np.float32(-self.dlt * self.dt)
        
        z    = KAOmother._KMAOcommu(phi,r,maxDlay,dlayIdx,self.eta,dlt_,self.fs,self.dt,kS_,omga_)

        #self.z = z
        profileFC   = self._fitFilterBands(z)
        fit, ccoef  = self.doDiffConnecti_SO(profileFC)
        orderG, orderGstd, orderL, orderLstd, order = self._doKuramotoOrder(z)
        self.log.append( [fit,ccoef,vel,kL,kG,orderG,orderGstd,orderL,orderLstd,order] )
        if self.lib == 1:
            return  np.array([fit])
        elif self.lib == 2:
            return fit
        else:
            raise ValueError('The library to optimize was not defined correctly')

class KAOsimpleSimuConstr(KAOmother):
    """ Local coupling is the same for all nodes """
    def get_name(self):
        return "KAOM with fix kL for all nodesl"
    
    def fitness(self,x):
        vel     = x[0]
        kG      = x[1]
        kL      = x[2]
        kS      = self.getAnatoCoupling(kG,kL)
        dlayStep, maxDlay = self.getDelays(vel)
        r, phi  = self._doNodeContainers(maxDlay)
        dlayIdx = self.doIndexDelay(r,dlayStep)
        
        # scales by dt, try to reduce floating point error, and speed-up
        kS_ = np.float32(kS*self.dt)
        omga_ = np.float32(self.omega*self.dt)
        dlt_ = np.float32(-self.dlt * self.dt)
        
        z    = KAOmother._KMAOcommu(phi,r,maxDlay,dlayIdx,self.eta,dlt_,self.fs,self.dt,kS_,omga_)

        #self.z = z
        profileFC   = self._fitFilterBands(z)
        fit, ccoef  = self.doDiffConnecti_SO(profileFC)
        orderG, orderGstd, orderL, orderLstd, order = self._doKuramotoOrder(z)
        if 0.1 > order > 0.6: # penalization if out of constrains
            fit = 0.5 * self.fBands.shape[0] * self.C ** 2
            ccoef = 0.0
        self.log.append( [fit,ccoef,vel,kL,kG,orderG,orderGstd,orderL,orderLstd,order] )
        if self.lib == 1:
            return  np.array([fit])
        elif self.lib == 2:
            return fit
        else:
            raise ValueError('The library to optimize was not defined correctly')
    
class kaoSimplMultiObj(KAOmother):
    """ Multiobjective optimization. Each frequency band is on objective function.
        Local coupling is the same for all nodes """
        
    def get_name(self):
        return "KAO simple, multi-objective optimization"
    
    def get_nobj(self):
        return self.fBands.shape[0]
    
    def fitness(self,x):
        vel     = x[0]
        kG      = x[1]
        kL      = x[2]
        kS      = self.getAnatoCoupling(kG,kL)
        dlayStep, maxDlay = self.getDelays(vel)
        r, phi  = self._doNodeContainers(maxDlay)
        dlayIdx = self.doIndexDelay(r,dlayStep)
        
        # scales by dt, try to reduce floating point error, and speed-up
        kS_ = np.float32(kS*self.dt)
        omga_ = np.float32(self.omega*self.dt)
        dlt_ = np.float32(-self.dlt * self.dt)
        
        z    = KAOmother._KMAOcommu(phi,r,maxDlay,dlayIdx,self.eta,dlt_,self.fs,self.dt,kS_,omga_)

        #self.z = z
        profileFC   = self._fitFilterBands(z)
        fit, ccoef  = self.doDiffConnecti_MO(profileFC)
        orderG, orderGstd, orderL, orderLstd, order = self._doKuramotoOrder(z)
        self.log.append( [fit,ccoef,vel,kL,kG,orderG,orderGstd,orderL,orderLstd,order] )       
        return fit

class kaoSimplMultiObjContr(KAOmother):
    """ Multiobjective optimization. Each frequency band is one objective function.
        It has two inequality
        constrains to bound the global order
    """
    def get_name(self):
        return "KAO simple, constrained multi-objective optimization"
    
    def get_nobj(self):
        return self.fBands.shape[0]
    
    def get_nic(self):
         """ Number of inequality constrains """
         return 2
    
    def fitness(self,x):
        """ Constrains for global order of type c_i <= 0 """
        vel     = x[0]
        kG      = x[1]
        kL      = x[2]
        kS      = self.getAnatoCoupling(kG,kL)
        dlayStep, maxDlay = self.getDelays(vel)
        r, phi  = self._doNodeContainers(maxDlay)
        dlayIdx = self.doIndexDelay(r,dlayStep)
        
        # scales by dt, try to reduce floating point error, and speed-up
        kS_ = np.float32(kS*self.dt)
        omga_ = np.float32(self.omega*self.dt)
        dlt_ = np.float32(-self.dlt * self.dt)
        
        z    = KAOmother._KMAOcommu(phi,r,maxDlay,dlayIdx,self.eta,dlt_,self.fs,self.dt,kS_,omga_)

        #self.z = z
        profileFC   = self._fitFilterBands(z)
        fit,ccoef   = self.doDiffConnecti_MO(profileFC)
        orderG, orderGstd, orderL, orderLstd, order1 = self._doKuramotoOrder(z)
        self.log.append( [fit,ccoef,vel,kL,kG,orderG,orderGstd,orderL,orderLstd,order1] )

        # pagmo:    [fitness , inequality 1, ienquality 2]
        # platypus: (fitness , [inequality 1, ienquality 2])
        constrains = [0.1 - order1, order1 - 0.6]
        if self.lib == 1:
            return np.append(fit, constrains)
        elif self.lib == 2:
            return (fit, constrains)
        else:
            raise ValueError('The library to optimize was not defined correctly')
            
class KAOnodes(KAOmother):
    def get_name(self):
        return "kao with all nodes"
    
    def get_bounds(self):
        """Boundaries on: velocity, kG, kL"""
        upbound     = self.upBound[0:2] + [self.upBound[2]] * self.C
        lowbound    = self.lowBound[0:2] + [self.lowBound[2]] * self.C
        return (lowbound, upbound)
    
    def fitness(self,x):
        vel     = x[0]
        kG      = x[1]
        kL      = x[2]
        kS      = self.getAnatoCoupling(kG,kL)
        dlayStep, maxDlay = self.getDelays(vel)
        r, phi  = self._doNodeContainers(maxDlay)
        dlayIdx = self.doIndexDelay(r,dlayStep)
        
        # scales by dt, try to reduce floating point error, and speed-up
        kS_ = np.float32(kS*self.dt)
        omga_ = np.float32(self.omega*self.dt)
        dlt_ = np.float32(-self.dlt * self.dt)
        
        z    = KAOmother._KMAOcommu(phi,r,maxDlay,dlayIdx,self.eta,dlt_,self.fs,self.dt,kS_,omga_)

        #self.z = z
        profileFC   = self._fitFilterBands(z)
        fit, ccoef  = self.doDiffConnecti_SO(profileFC)
        orderG, orderGstd, orderL, orderLstd, order = self._doKuramotoOrder(z)
        self.log.append( [fit,ccoef,vel,kL,kG,orderG,orderGstd,orderL,orderLstd, order] )
        if self.lib == 1:
            return  np.array([fit])
        elif self.lib == 2:
            return fit
        else:
            raise ValueError('The library to optimize was not defined correctly')

class KAOnodesSimuConstr(KAOmother):
    def get_name(self):
        return "kao with all nodes"
    
    def get_bounds(self):
        """Boundaries on: velocity, kG, kL"""
        upbound     = self.upBound[0:2] + [self.upBound[2]] * self.C
        lowbound    = self.lowBound[0:2] + [self.lowBound[2]] * self.C
        return (lowbound, upbound)
    
    def fitness(self,x):
        vel     = x[0]
        kG      = x[1]
        kL      = x[2]
        kS      = self.getAnatoCoupling(kG,kL)
        dlayStep, maxDlay = self.getDelays(vel)
        r, phi  = self._doNodeContainers(maxDlay)
        dlayIdx = self.doIndexDelay(r,dlayStep)
        
        # scales by dt, try to reduce floating point error, and speed-up
        kS_ = np.float32(kS*self.dt)
        omga_ = np.float32(self.omega*self.dt)
        dlt_ = np.float32(-self.dlt * self.dt)
        
        z    = KAOmother._KMAOcommu(phi,r,maxDlay,dlayIdx,self.eta,dlt_,self.fs,self.dt,kS_,omga_)

        #self.z = z
        profileFC   = self._fitFilterBands(z)
        fit, ccoef  = self.doDiffConnecti_SO(profileFC)
        orderG, orderGstd, orderL, orderLstd, order = self._doKuramotoOrder(z)
        if 0.1 > order > 0.6: # penalization if out of constrains
            fit = 0.5 * self.fBands.shape[0] * self.C ** 2
            ccoef = 0.0
        self.log.append( [fit,ccoef,vel,kL,kG,orderG,orderGstd,orderL,orderLstd, order] )
        if self.lib == 1:
            return  np.array([fit])
        elif self.lib == 2:
            return fit
        else:
            raise ValueError('The library to optimize was not defined correctly')

    
class KAOnodesMultiObj(KAOmother):
    """ KM model with AO reduction. Optimized delay, global, and local coupling.
        Each frequency band is one cost function, to be solved as multi-objective 
        optimization problem."""    
        
    def get_name(self):
        return "kao with all nodes, multiobjective optimization"
    
    def get_nobj(self):
        return self.fBands.shape[0]
    
    def get_bounds(self):
        """Boundaries on: velocity, kG, kL"""
        upbound     = self.upBound[0:2] + [self.upBound[2]] * self.C
        lowbound    = self.lowBound[0:2] + [self.lowBound[2]] * self.C
        return (lowbound, upbound)

    def fitness(self,x):
        vel     = x[0]
        kG      = x[1]
        kL      = x[2:]
        kS      = self.getAnatoCoupling(kG,kL)
        dlayStep, maxDlay = self.getDelays(vel)
        r, phi  = self._doNodeContainers(maxDlay)
        dlayIdx = self.doIndexDelay(r,dlayStep)
        
        # scales by dt, try to reduce floating point error, and speed-up
        kS_ = np.float32(kS*self.dt)
        omga_ = np.float32(self.omega*self.dt)
        dlt_ = np.float32(-self.dlt * self.dt)
        
        z    = KAOmother._KMAOcommu(phi,r,maxDlay,dlayIdx,self.eta,dlt_,self.fs,self.dt,kS_,omga_)

        #self.z = z
        profileFC   = self._fitFilterBands(z)
        fit, ccoef  = self.doDiffConnecti_MO(profileFC)
        orderG, orderGstd, orderL, orderLstd, order = self._doKuramotoOrder(z)
        self.log.append( [fit,ccoef,vel,kL,kG,orderG,orderGstd,orderL,orderLstd,order] )
        return fit
    
class KAOnodesMultiObjConstr(KAOmother):
    """ KM model with AO reduction. Optimized delay, global, and local coupling.
        Each frequency band is one cost function, to be solved as multi-objective 
        optimization problem. It has two inequality constrains to bound the global order
    """
    def get_name(self):
        return "kao with all nodes, multiobjective optimization with constrains"
    
    def get_nobj(self):
        return self.fBands.shape[0]
 
    def get_nic(self):
         """ Number of inequality constrains """
         return 2    
    
    def get_bounds(self):
        """Boundaries on: velocity, kG, kL"""
        upbound     = self.upBound[0:2] + [self.upBound[2]] * self.C
        lowbound    = self.lowBound[0:2] + [self.lowBound[2]] * self.C
        return (lowbound, upbound)
    
    def fitness(self,x):
        """ Constrains for global order of type c_i <= 0 """
        vel     = x[0]
        kG      = x[1]
        kL      = x[2:]
        kS      = self.getAnatoCoupling(kG,kL)
        dlayStep, maxDlay = self.getDelays(vel)
        r, phi  = self._doNodeContainers(maxDlay)
        dlayIdx = self.doIndexDelay(r,dlayStep)
        
        # scales by dt, try to reduce floating point error, and speed-up
        kS_ = np.float32(kS*self.dt)
        omga_ = np.float32(self.omega*self.dt)
        dlt_ = np.float32(-self.dlt * self.dt)
        
        z    = KAOmother._KMAOcommu(phi,r,maxDlay,dlayIdx,self.eta,dlt_,self.fs,self.dt,kS_,omga_)

        #self.z = z
        profileFC   = self._fitFilterBands(z)
        fit, ccoef  = self.doDiffConnecti_MO(profileFC)
        orderG, orderGstd, orderL, orderLstd, order1 = self._doKuramotoOrder(z)
        self.log.append( [fit,ccoef,vel,kL,kG,orderG,orderGstd,orderL,orderLstd,order1] )

        # pagmo:    [fitness , inequality 1, ienquality 2]
        # platypus: (fitness , [inequality 1, ienquality 2])
        constrains = [0.1 - order1, order1 - 0.6]
        if self.lib == 1:
            return np.append(fit, constrains)
        elif self.lib == 2:
            return (fit, constrains)
        else:
            raise ValueError('The library to optimize was not defined correctly')