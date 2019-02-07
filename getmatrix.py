import numpy as np
from simulation import *
from model import *
from common import *
from time import time

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
    colli[sim.mol.lcu, sim.mol.lcl] = sim.mol.down[:,idx]
    colli[sim.mol.lcl, sim.mol.lcu] = sim.mol.up[:,idx]

    if sim.mol.part2id:
        colli2[sim.mol.lcu2, sim.mol.lcl2] = sim.mol.down2[:,idx]
        colli2[sim.mol.lcl2, sim.mol.lcu2] = sim.mol.up2[:,idx]


    # Sum by rows and add the conservation equation to the
    # left-hand side (at position nlev+1).
    ctot = np.sum(colli, axis=1)

    if sim.mol.part2id:
        ctot2 = np.sum(colli2, axis=1)


    # Fill the rate matrix with the collisional contributions.
    # check if two second collisional partner exists in model
    if 'ne' in sim.model.grid:
        mtrx[np.diag_indices(sim.nlev)] += sim.model.grid['nh2'][idx]*ctot + sim.model.grid['ne'][idx]*ctot2
        mtrx[:-1,:-1] += -sim.model.grid['nh2'][idx]*colli.T - sim.model.grid['ne'][idx]*colli2.T
        mtrx[sim.nlev,:-1] = 1.
        mtrx[:-1,sim.nlev] = 0.
    else:
        mtrx[np.diag_indices(sim.nlev)] += sim.model.grid['nh2'][idx]*ctot
        mtrx[:-1,:-1] += -sim.model.grid['nh2'][idx]*colli.T
        mtrx[sim.nlev,:-1] = 1.
        mtrx[:-1,sim.nlev] = 0.

    return mtrx
