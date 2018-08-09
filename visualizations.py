#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 28 17:25:34 2018

@author: oscar
"""
import numpy as np
import scipy.signal as sg
import matplotlib 
matplotlib.use('TkAgg')
import matplotlib.pyplot as pl
from mpl_toolkits.mplot3d import Axes3D
import pygmo as po


##
#pathdata = '/Users/p277634/python/kaoModel/optimResult/'
#filename = 'pso_nodes'
#extdata  = '.npz'
##extfig   = '.pdf'
#toLoad   = pathdata + filename + extdata
##
#with np.load(toLoad) as data:
#    vel     = data['velocity']
#    kL      = data['kL']
#    kG      = data['kG']
#    fit     = data['fitness']
#    kordL   = data['KordrL']
#    kordG   = data['KordrG']
#    kordLstd= data['KordrLstd']
#    kordLstd= data['KordrLstd']
##    
#vel90   = np.tile(vel,(90,1)).T.reshape((1,90*fit.shape[0]))
#kL90    = kL.reshape((1,90*fit.shape[0]))
#kG90    = np.tile(kG,(90,1)).T.reshape((1,90*fit.shape[0]))
#fit90   = np.tile(fit,(90,1)).T.reshape((1,90*fit.shape[0]))
###
vel90   = np.tile(velocity,(90,1)).T.reshape((1,90*fitness.shape[0]))
kL90    = kL.reshape((1,90*fitness.shape[0]))
kG90    = np.tile(kG,(90,1)).T.reshape((1,90*fitness.shape[0]))
fit90   = np.tile(fitness,(90,1)).T.reshape((1,90*fitness.shape[0]))

#bestIx = np.argmin(fitness)
#best = {'kL':kL[bestIx].round(1),'kG':kG[bestIx].round(1),'vel':vel[bestIx].round(1)}

def doScatter3D(vel, kL, kG, fit, title):
    fig = pl.figure()
    ax = fig.add_subplot(111, projection='3d')
    sc = ax.scatter(kL, kG, vel, c=fit, cmap="hot")
    ax.grid(True)
    ax.set_xlabel('Local Coupling')
    ax.set_ylabel('Global Coupling')
    ax.set_zlabel('Velocity')
    #title = 'Best, velocity: ' + str(best['vel'])+ '[m/s]; kL: ' + str(best['kL']) + '; kG: ' + str(best['kG'])
    ax.set_title(title)
    clb = pl.colorbar(sc)
    clb.ax.set_title('  fitness')
    pl.show() 
    
def do3subScatterkLkGvel(vel, kL, kG, fit, title):
    fig = pl.figure()
    
    ax = fig.add_subplot(131)
    sc = ax.scatter(kL, kG, s=0.2, c=fit.ravel(), cmap="hot")
    ax.grid(True)
    ax.set_xlabel('Local Coupling')
    ax.set_ylabel('Global Coupling')
    ax.set_title(title)
    clb = pl.colorbar(sc)
    clb.ax.set_title('  fitness')
    
    ax = fig.add_subplot(132)
    sc = ax.scatter(kL, vel, s=0.2, c=fit.ravel(), cmap="hot")
    ax.grid(True)
    ax.set_xlabel('Local Coupling')
    ax.set_ylabel('velocity')
    ax.set_title(title)
    clb = pl.colorbar(sc)
    clb.ax.set_title('  fitness')
    
    ax = fig.add_subplot(133)
    sc = ax.scatter(kG, vel, s=0.2, c=fit.ravel(), cmap="hot")
    ax.grid(True)
    ax.set_xlabel('Global Coupling')
    ax.set_ylabel('velocity')
    ax.set_title(title)
    clb = pl.colorbar(sc)
    clb.ax.set_title('  fitness')
    
    pl.show()
    

def do2subScatterGLorder(Gord, GordSD, Lord, LordSD, fit, title):
    fig = pl.figure()
    
    ax = fig.add_subplot(121)
    sc = ax.scatter(Gord, GordSD, s=0.2, c=fit, cmap="hot")
    ax.grid(True)
    ax.set_xlabel('Global Order')
    ax.set_ylabel('Global Order SD')
    ax.set_title(title)
    clb = pl.colorbar(sc)
    clb.ax.set_title('  fitness')
    
    ax = fig.add_subplot(122)
    sc = ax.scatter(Lord, LordSD, s=0.2, c=fit, cmap="hot")
    ax.grid(True)
    ax.set_xlabel('Local Order')
    ax.set_ylabel('Local Order SD')
    ax.set_title(title)
    clb = pl.colorbar(sc)
    clb.ax.set_title('  fitness')
    
    pl.show()

def doScatter2DkLkG(vel, kL, kG, fit, title):
    fig = pl.figure()
    ax = fig.add_subplot(111)
    sc = ax.scatter(kL, kG, s=vel, c=fit, cmap="hot")
    ax.grid(True)
    ax.set_xlabel('Local Coupling')
    ax.set_ylabel('Global Coupling')
    #title = 'Best, velocity (size): ' + str(best['vel'])+ '[m/s]; kL: ' + str(best['kL']) + '; kG: ' + str(best['kG'])
    ax.set_title(title)
    clb = pl.colorbar(sc)
    clb.ax.set_title('  fitness')
    pl.show()    

def doScatter2DkLvel(vel, kL, kG, fit, title):
    fig = pl.figure()
    ax = fig.add_subplot(111)
    sc = ax.scatter(kL, vel, s=kG/np.mean(kG), c=fit, cmap="hot")
    ax.grid(True)
    ax.set_xlabel('Local Coupling')
    ax.set_ylabel('velocity')
    #title = 'Best, velocity: ' + str(best['vel'])+ '[m/s]; kL: ' + str(best['kL']) + '; kG (size): ' + str(best['kG'])
    ax.set_title(title)
    clb = pl.colorbar(sc)
    clb.ax.set_title('  fitness')
    pl.show()  

def doScatter2DkGvel(vel, kL, kG, fit, title):
    fig = pl.figure()
    ax = fig.add_subplot(111)
    sc = ax.scatter(kG, vel, s=kL/np.mean(kL), c=fit, cmap="hot")
    ax.grid(True)
    ax.set_xlabel('Global Coupling')
    ax.set_ylabel('velocity')
    #title = 'Best, velocity: ' + str(best['vel'])+ '[m/s]; kL (size): ' + str(best['kL']) + '; kG: ' + str(best['kG'])
    ax.set_title(title)
    clb = pl.colorbar(sc)
    clb.ax.set_title('  fitness')
    pl.show()
    
def doScatter2DkL(vel, kL, kG, fit, title):
    fig = pl.figure()
    ax = fig.add_subplot(111)
    sc = ax.scatter(kL, s=kG/np.mean(kG), c=fit, cmap="hot")
    ax.grid(True)
    ax.set_xlabel('Global Coupling')
    ax.set_ylabel('velocity')
    #title = 'Best, velocity: ' + str(best['vel'])+ '[m/s]; kL (size): ' + str(best['kL']) + '; kG: ' + str(best['kG'])
    ax.set_title(title)
    clb = pl.colorbar(sc)
    clb.ax.set_title('  fitness')
    pl.show()

def doFitGenrations(fit):   
    fig = pl.figure()
    ax = fig.add_subplot(111)
    ax.plot(-1*fit)
    ax.grid(True)
    ax.set_xlabel('genrations | iterations')
    ax.set_ylabel('Fitness')
    title = 'Fitness/correation to empirical data'
    ax.set_title(title)
    pl.show()
    
def doFuncionalConnectivty(z, fBands, fs, tMin):
    coeFil, coeFilEnv = desingFilterBands(fBands,fs)
    connecti = connectvityBands(z, coeFil, coeFilEnv, tMin, fs)
    fig, axes = pl.subplots(2, int(np.ceil(fBands.shape[0]/2)))#,sharex=True,sharey=True)
    axes = axes.ravel()
    for ix in range(fBands.shape[0]):
        ms = axes[ix].matshow(connecti[:,:,ix], origin='lower', cmap="coolwarm", vmin=-1, vmax=1)
        title = 'frequencies: ' + str(fBands[ix,:]) + ' Hz' 
        axes[ix].set_title(title)
    pl.colorbar(ms)
    fig.show()    
    
def desingFilterBands(fBands,fs):
    nyq   = fs / 2.0
    trans = 2.0
    coeFil = []
    for freq in fBands:
        # Filter frequency bands
        passCut = freq / nyq
        stopCut = [(freq[0] - trans) / nyq, (freq[1] + trans) / nyq]
        coeFil.append(sg.iirdesign(passCut, stopCut, gpass=0.0025, gstop=30.0,
                                        analog=False, ftype='cheby2', output='sos'))
    #coeFil = np.dstack(coeFil)
    # Filter envelops
    coeFilEnv = sg.iirdesign(0.5 / nyq, (0.5+trans)/nyq , gpass=0.0025, gstop=30.0,
                                        analog=False, ftype='cheby2', output='sos')
    return coeFil, coeFilEnv
    
    
def connectvityBands(z, coeFil, coeFilEnv, tMin, fs):
    envCo = []
    for coefsos in coeFil:
        # filter frequency bands
        zFilt   = sg.sosfiltfilt(coefsos, np.imag(z), axis=1, padtype='odd')
        zEnv    = np.abs(sg.hilbert(zFilt, axis=1))
        # filter envelope
        zEnvFilt= sg.sosfiltfilt(coeFilEnv, zEnv, axis=1, padtype='odd')
        # Correlation discarding warmup time
        envCo.append(np.corrcoef(zEnvFilt[:,int(tMin*fs):-int(tMin*fs)], 
                                 rowvar=True))
    envCo = np.dstack(envCo)
    return envCo

def simpleConnecti(connecti,fBands):
    fig, axes = pl.subplots(2, int(np.ceil(fBands.shape[0]/2)))#,sharex=True,sharey=True)
    axes = axes.ravel()
    for ix in range(fBands.shape[0]):
        ms = axes[ix].matshow(connecti[ix,:,:], origin='lower', cmap="coolwarm", vmin=-1, vmax=1)
        title = 'frequencies: ' + str(fBands[ix,:]) + ' Hz' 
        axes[ix].set_title(title)
    pl.colorbar(ms)
    fig.show()
    
def dokLkGvelMultiObj(vel, kL, kG, fit, title):
    fBands  = 4
    locSub  = np.range(1,3*(fBands+1)+1).reshape(3,fBands+1)
    fit_    = np.zeros(fBands+1,vel.shape[0])
    for i in range(vel.shape[0]):
        for j in range(fBands):    
            fit_[i,j] = fit[i][j]

    fit_[:,-1]= getHypervolumenMOO(fit)
    
    fig = pl.figure()
    for i in range(fBands):
   
        ax = fig.add_subplot(3,5,locSub[0, i])
        sc = ax.scatter(kL, kG, s=0.2, c=fit_[:,i], cmap="hot")
        ax.grid(True)
        ax.set_xlabel('Local Coupling')
        ax.set_ylabel('Global Coupling')
        ax.set_title(title)
        clb = pl.colorbar(sc)
        clb.ax.set_title('  fitness')
        
        ax = fig.add_subplot(3,5,locSub[1, i])
        sc = ax.scatter(kL, vel, s=0.2, c=fit_[:,i], cmap="hot")
        ax.grid(True)
        ax.set_xlabel('Local Coupling')
        ax.set_ylabel('velocity')
        ax.set_title(title)
        clb = pl.colorbar(sc)
        clb.ax.set_title('  fitness')
        
        ax = fig.add_subplot(3,5,locSub[2, i])
        sc = ax.scatter(kG, vel, s=0.2, c=fit_[:,i], cmap="hot")
        ax.grid(True)
        ax.set_xlabel('Global Coupling')
        ax.set_ylabel('velocity')
        ax.set_title(title)
        clb = pl.colorbar(sc)
        clb.ax.set_title('  fitness')
    
    pl.show()
    
def getHypervolumenMOO(fit):
    sd = np.zeros(fit.shape)
    for i in range(fit.shape[0]):
        sd[i] = po.hypervolume(fit[i][None]).compute([1,1,1,1])
    return sd