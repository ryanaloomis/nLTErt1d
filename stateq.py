import numpy as np
# from simulation import *
# from model import *
from common import eps
from getjbar import getjbar
from getmatrix import getmatrix
# from scipy.linalg import lu_factor, lu_solve
# from time import time


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

        # fill collision rate matrix
        if (debug):
            print('[debug] calling getmatrix, iter= ' + str(iter))
        ratem = getmatrix(sim, idx)

        # solve rate matrix
        newpop = np.linalg.lstsq(ratem, newpop, rcond=None)[0]

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
