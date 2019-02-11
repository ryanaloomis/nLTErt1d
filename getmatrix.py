import numpy as np
from simulation import *
from model import *
from common import *
from time import time
from numba import jit

@jit(nopython=True)
def getmatrix(part2id, phot, nh2, ne, lau, lal, lcu, lcl, lcu2, lcl2, down, up, down2, up2, aeinst, beinstu, beinstl, nline, nlev, jbar, idx):
    mtrx = np.zeros((nlev+1, nlev+1))
    colli = np.zeros((nlev, nlev))
    colli2 = np.zeros((nlev, nlev))

    for t in range(nline):
        k = lau[t]                                          
        l = lal[t]                                           
        mtrx[k,k] += beinstu[t]*jbar[t] + aeinst[t] 
        mtrx[l,l] += beinstl[t]*jbar[t]                     
        mtrx[k,l] += -beinstl[t]*jbar[t]                    
        mtrx[l,k] += -beinstu[t]*jbar[t] - aeinst[t]


    # create collision rate matrix
    for i in range(nlev):
        colli[lcu[i], lcl[i]] = down[i,idx]
        colli[lcl[i], lcu[i]] = up[i,idx]

    if part2id:
        for i in range(nlev):
            colli2[lcu2[i], lcl2[i]] = down2[i,idx]
            colli2[lcl2[i], lcu2[i]] = up2[i,idx]


    # Sum by rows and add the conservation equation to the
    # left-hand side (at position nlev+1).
    ctot = np.sum(colli, axis=1)

    if part2id:
        ctot2 = np.sum(colli2, axis=1)


    # Fill the rate matrix with the collisional contributions.
    # check if two second collisional partner exists in model
    if part2id:
        for i in range(nlev):
            mtrx[i,i] += nh2[idx]*ctot[i] + ne[idx]*ctot2[i]
        mtrx[:-1,:-1] += -nh2[idx]*colli.T - ne[idx]*colli2.T
        mtrx[nlev,:-1] = 1.
        mtrx[:-1,nlev] = 0.
    else:
        for i in range(nlev):
            mtrx[i,i] += nh2[idx]*ctot[i]
        mtrx[:-1,:-1] += -nh2[idx]*colli.T
        mtrx[nlev,:-1] = 1.
        mtrx[:-1,nlev] = 0.

    return mtrx
