import numpy as np
from simulation import *
from model import *
from common import *
from time import time

def getjbar(sim, idx, debug):
    vsum = 0.
    sim.mol.jbar = np.zeros(sim.nline)

    jnu_dust = np.zeros(sim.nline)
    alpha_dust = np.zeros(sim.nline)

    jnu_dust = sim.dust[:, idx]*sim.knu[:, idx]
    alpha_dust = sim.knu[:, idx]

    ds = sim.phot[0]
    vfac = sim.phot[1]
    vsum = np.sum(vfac)

    jnu = jnu_dust[:,np.newaxis] + vfac[np.newaxis,:]/sim.model.grid['doppb'][idx]*hpip*sim.model.grid['nmol'][idx]*(sim.pops[sim.mol.lau, idx])[:,np.newaxis]*sim.mol.aeinst[:,np.newaxis]
    alpha = alpha_dust[:,np.newaxis] + vfac[np.newaxis,:]/sim.model.grid['doppb'][idx]*hpip*sim.model.grid['nmol'][idx]*((sim.pops[sim.mol.lal,idx])[:,np.newaxis]*sim.mol.beinstl[:,np.newaxis] - (sim.pops[sim.mol.lau,idx])[:,np.newaxis]*sim.mol.beinstu[:,np.newaxis])

    snu = jnu/alpha/sim.norm[:,np.newaxis]
    snu[np.abs(alpha) < eps] = 0.

    tau = alpha*ds[np.newaxis,:]
    tau[tau < negtaulim] = negtaulim

    sim.mol.jbar = np.sum(vfac[np.newaxis,:]*(np.exp(-tau)*sim.phot[2:,:] + (1 - np.exp(-tau))*snu), axis=1)

    if (vsum > 0.):
        sim.mol.jbar *= sim.norm/vsum # Normalize and scale by norm and vsum

    return
