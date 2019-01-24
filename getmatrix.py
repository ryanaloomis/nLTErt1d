import numpy as np
from simulation import simulation
from model import model
from common import *

def getmatrix(sim, idx):
    mtrx = np.zeros((sim.nlev+1, sim.nlev+1))
    colli = np.zeros((sim.nlev, sim.nlev))
    colli2 = np.zeros((sim.nlev, sim.nlev))
    
    for t in range(sim.nline):
        k = sim.mol.lau[t]                                          
        l = sim.mol.lal[t]                                           
        mtrx[k,k] += sim.mol.beinstu[t]*sim.mol.jbar[t] + sim.mol.aeinst[t] 
        mtrx[l,l] += sim.mol.beinstl[t]*sim.mol.jbar[t]                     
        mtrx[k,l] += -sim.mol.beinstl[t]*sim.mol.jbar[t]                    
        mtrx[l,k] += -sim.mol.beinstu[t]*sim.mol.jbar[t] - sim.mol.aeinst[t]
    

    # create collision rate matrix

    for t in range(sim.ntrans):
        colli[sim.mol.lcu[t], sim.mol.lcl[t]] = sim.mol.down_loc[t]
        colli[sim.mol.lcl[t], sim.mol.lcu[t]] = sim.mol.up_loc[t]

    for t in range(sim.ntrans2):
        colli2[sim.mol.lcu2[t], sim.mol.lcl2[t]] = sim.mol.down2_loc[t]
        colli2[sim.mol.lcl2[t], sim.mol.lcu2[t]] = sim.mol.up2_loc[t]


    # Sum by rows and add the conservation equation to the
    # left-hand side (at position nlev+1).

    for s in range(sim.nlev):
        ctot[s] = 0.
        for t in range(sim.nlev):
            ctot[s] += colli[s,t]

    for s in range(sim.nlev):
        ctot2[s] = 0.
        for t in range(sim.nlev):
            ctot2[s] += colli2[s,t]


    # Fill the rate matrix with the collisional contributions.
    
    for s in range(sim.nlev):
        mtrx[s,s] = mtrx[s,s] + sim.model.grid['nh2'][idx]*ctot[s] + sim.model.grid['ne'][idx]*ctot2[s]
        for t in range(sim.nlev):
            mtrx[s,t] += -sim.model.grid['nh2'][idx]*colli[t,s] - sim.model.grid['ne'][idx]*colli2[t,s]
        mtrx[sim.nlev, s] = 1.
        mtrx[s, sim.nlev] = 0.


    return mtrx
