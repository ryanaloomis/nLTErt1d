import numpy as np
from simulation import simulation
from model import model
from common import *

def getjbar(sim, idx, debug):
    vsum = 0.
    sim.mol.jbar = np.zeros(sim.nline)

    for iline in range(sim.nline):
        jnu_dust[iline] = sim.dust[iline, idx]*sim.knu[iline, idx] # TODO
        alpha_dust[iline] = sim.knu[iline, idx] # TODO

    for iphot in range(sim.nphot[idx]):
        if (debug): print('[debug] iphot = ' + str(iphot))

        ds = phot[0, iphot]     # TODO
        vfac = phot[1, iphot]   # TODO

        for iline in range(sim.nline):
            jnu = jnu_dust[iline] + vfac/sim.model.grid['doppb'][idx]*hpip*sim.model.grid['nmol'][idx]*sim.pops[sim.mol.lau[iline],idx]*sim.mol.aeinst[iline] # TODO

            alpha = alpha_dust[iline] + vfac/sim.model.grid['doppb'][idx]*hpip*sim.model.grid['nmol'][idx]*(sim.pops[sim.mol.lal[iline],idx]*sim.mol.beinstl[iline] - sim.pops[lau[iline],idx]*sim.mol.beinstu[iline]) # TODO

            if (np.abs(alpha) < eps):
                snu = 0.
            else:
                snu = jnu/alpha/norm[iline] # TODO

            dtau = alpha*ds
            if (dtau < negtaulim): # Limit negative opacity
                dtau = negtaulim

            # Add intensity along line segment
            sim.mol.jbar[iline] += vfac*(np.exp(-tau)*phot[iline+2, iphot] + (1 - np.exp(-tau))*snu) # TODO

        vsum += vfac

        if (vsum > 0.):
            for iline in range(sim.nline):
                sim.mol.jbar[iline] *= norm[iline]/vsum # Normalize and scale by norm and vsum

    return  # TODO
