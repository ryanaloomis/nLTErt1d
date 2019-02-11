import numpy as np
from common import eps
from getjbar import getjbar
from getmatrix import getmatrix
from time import time
from numba import jit


@jit(nopython=True)
def solvematrix(ratem, newpop):
    result = np.linalg.lstsq(ratem, newpop)
    return result[0]
  

def stateq(sim, idx, debug=False, miniter=10, maxiter=100, tol=1e0-6):
    """
    Iterate for convergence between radiation field and excitation.

    Args:
        sim (simulation instance): Instance of the simulation.
        idx (int): Radial index.
        debug (optional[bool]): Print debug messages.
        miniter (optional[int]): Minimum number of iterations.
        maxiter (optional[int]): Maximum number of iterations.
        tol (optional[float]): Tolerate of convergence.

    Returns:
        stater (float): Difference between the values.
    """

    opop = np.zeros(sim.nlev)
    oopop = np.zeros(sim.nlev)

    # Get updated jbar.
    for iter in range(maxiter):
        if debug:
            print('[debug] calling getjbar, iter= ' + str(iter))
        getjbar(sim, idx, debug)

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
        opop = sim.pops[:, idx]
        sim.pops[:, idx] = newpop[:-1]
        if np.any((np.minimum(newpop[:-1], oopop) > sim.minpop)):
            _diff = np.maximum(np.abs(newpop[:-1] - opop) / newpop[:-1],
                               np.abs(newpop[:-1] - oopop) / newpop[:-1])
            diff = np.maximum(_diff, diff)

        if (iter > miniter) and (diff < tol):
            continue

    # If not all converged, save diff in staterr
    if diff > tol:
        sim.staterr = diff
    return sim.staterr
