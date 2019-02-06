import numpy as np
from simulation import *
from model import *
from common import *

def getjbar(sim, idx, debug):
    vsum = 0.
    sim.mol.jbar = np.zeros(sim.nline)

    jnu_dust = np.zeros(sim.nline)
    alpha_dust = np.zeros(sim.nline)

    for iline in range(sim.nline):
        jnu_dust[iline] = sim.dust[iline, idx]*sim.knu[iline, idx]
        alpha_dust[iline] = sim.knu[iline, idx]

    for iphot in range(sim.nphot[idx]):
        if (debug): print('[debug] iphot = ' + str(iphot))

        ds = sim.phot[0, iphot]
        vfac = sim.phot[1, iphot]

        for iline in range(sim.nline):
            jnu = jnu_dust[iline] + vfac/sim.model.grid['doppb'][idx]*hpip*sim.model.grid['nmol'][idx]*sim.pops[sim.mol.lau[iline],idx]*sim.mol.aeinst[iline]

            alpha = alpha_dust[iline] + vfac/sim.model.grid['doppb'][idx]*hpip*sim.model.grid['nmol'][idx]*(sim.pops[sim.mol.lal[iline],idx]*sim.mol.beinstl[iline] - sim.pops[sim.mol.lau[iline],idx]*sim.mol.beinstu[iline])

            if (np.abs(alpha) < eps):
                snu = 0.
            else:
                snu = jnu/alpha/sim.norm[iline]

            tau = alpha*ds
            if (tau < negtaulim): # Limit negative opacity
                tau = negtaulim
            
            # Add intensity along line segment
            sim.mol.jbar[iline] += vfac*(np.exp(-tau)*sim.phot[iline+2, iphot] + (1 - np.exp(-tau))*snu)

        vsum += vfac

    if (vsum > 0.):
        for iline in range(sim.nline):
            sim.mol.jbar[iline] *= sim.norm[iline]/vsum # Normalize and scale by norm and vsum

    return
