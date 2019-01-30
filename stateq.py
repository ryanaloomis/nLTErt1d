import numpy as np
from simulation import *
from model import *
from common import *
from getjbar import *

miniter = 10
maxiter = 100
tol = 1.e-6


def stateq(sim, idx, debug):
    # do something to load collision rates # TODO

    

    # Iterate for convergence between radiation field and excitation
    opop = np.zeros(sim.nlev)
    oopop = np.zeros(sim.nlev)
    
    for iter in range(maxiter):
        # get updated jbar
        if (debug): print('[debug] calling getjbar, iter= ' + str(iter))
        getjbar(sim, idx, debug) # TODO

        newpop = np.zeros(sim.nlev + 1)
        newpop[-1] = 1.

        # fill collision rate matrix
        if (debug): print('[debug] calling getmatrix, iter= ' + str(iter))
        ratem = getmatrix(sim, idx)

        # solve with LU-decomposition
        if (debug): print('[debug] calling ludcmp, iter= ' + str(iter))
        ludcmp(ratem,sim.nlev+1,sim.maxlev+1,indx,d) # TODO
        if (debug): print('[debug] calling ludcmp, iter= ' + str(iter))
        lubksb(ratem,sim.nlev+1,sim.maxlev+1,indx,newpop) # TODO

        
        numb = 0
        diff = 0
        for s in range(sim.nlev):
            newpop[s] = np.max([newpop[s], eps])
            oopop[s] = opop[s]
            sim.pops[s, idx] = newpop[s]
            if (np.min([newpop[s], opop[s], oopop[s]]) > minpop):
                numb += 1
                diff = np.max([np.abs(newpop[s] - opop[s])/newpop[s], np.abs(newpop[s] - oopop[s])/newpop[s], diff])

        if (iter > miniter) and (diff < tol): continue

    # If not all converged, save diff in staterr
    if (diff > tol): staterr = diff

    return staterr


