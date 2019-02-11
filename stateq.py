import numpy as np
from simulation import *
from model import *
from common import *
from getjbar import *
from getmatrix import *
from time import time
from numba import jit

miniter = 10
maxiter = 100
tol = 1.e-6

@jit(nopython=True)
def solvematrix(ratem, newpop):
    result = np.linalg.lstsq(ratem, newpop)
    return result[0]


def stateq(sim, idx, debug):
    # Iterate for convergence between radiation field and excitation
    opop = np.zeros(sim.nlev)
    oopop = np.zeros(sim.nlev)
    
    jnu_dust = np.zeros(sim.nline)
    alpha_dust = np.zeros(sim.nline)

    jnu_dust = sim.dust[:, idx]*sim.knu[:, idx]
    alpha_dust = sim.knu[:, idx]

    ds = sim.phot[0]
    vfac = sim.phot[1]
    vsum = np.sum(vfac)

    for iter in range(maxiter):
        # get updated jbar
        #t0 = time()
        sim.mol.jbar = getjbar(ds, vfac, vsum, sim.phot, sim.model.grid['nmol'], sim.model.grid['doppb'], sim.mol.lau, sim.mol.lal, sim.mol.aeinst, sim.mol.beinstu, sim.mol.beinstl, sim.nline, sim.pops, jnu_dust, alpha_dust, sim.norm, idx)
        #t1 = time()
        #print "getjbar " + str((t1-t0)*100)

        newpop = np.zeros(sim.nlev + 1)
        newpop[-1] = 1.

        #t0 = time()
        # fill collision rate matrix
        if sim.mol.part2id:
            ne = sim.model.grid['ne']
            lcu2 = sim.mol.lcu2
            lcl2 = sim.mol.lcl2
            down2 = sim.mol.down2
            up2 = sim.mol.up2
        else:
            ne = np.zeros(sim.model.grid['nh2'].shape)
            lcu2 = np.zeros(sim.mol.lcu.shape).astype(int)
            lcl2 = np.zeros(sim.mol.lcl.shape).astype(int)
            down2 = np.zeros(sim.mol.down.shape)
            up2 = np.zeros(sim.mol.up.shape)

        ratem = getmatrix(bool(sim.mol.part2id), sim.phot, sim.model.grid['nh2'], ne, sim.mol.lau, sim.mol.lal, sim.mol.lcu, sim.mol.lcl, lcu2, lcl2, sim.mol.down, sim.mol.up, down2, up2, sim.mol.aeinst, sim.mol.beinstu, sim.mol.beinstl, sim.nline, sim.nlev, sim.mol.jbar, idx)
        #t1 = time()
        #print "getmatrix " + str((t1-t0)*100)

        #t0 = time()
        # solve rate matrix
        newpop = solvematrix(ratem, newpop)
        #t1 = time()
        #print "solve " + str((t1-t0)*100)

        diff = 0
        newpop = np.maximum(newpop, eps)
        oopop = opop
        opop = sim.pops[:,idx]
        sim.pops[:,idx] = newpop[:-1] 
        if np.any((np.minimum(newpop[:-1], oopop) > sim.minpop)):
            diff = np.max(np.maximum(np.maximum(np.abs(newpop[:-1] - opop)/newpop[:-1], np.abs(newpop[:-1] - oopop)/newpop[:-1]), diff))

        if (iter > miniter) and (diff < tol): continue

    # If not all converged, save diff in staterr
    if (diff > tol): sim.staterr = diff

    return sim.staterr


