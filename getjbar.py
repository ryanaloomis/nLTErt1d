from common import eps, negtaulim
import numpy as np
from simulation import *
from model import *
from common import *
from time import time
from numba import jit

@jit(nopython=True)
def getjbar(ds, vfac, vsum, phot, nmol, doppb, lau, lal, aeinst, beinstu, beinstl, nline, pops, jnu_dust, alpha_dust, norm, idx):
    """
    Set the jbar value for the given radial index.

    Args:
        sim (simulation instance):
        idx (int): Ind
        debug (optional[bool]): If True, print debug messages.

    Returns:
        jbar
    """
    jbar = np.zeros(nline)

    jnu_precalc = 1./doppb[idx]*hpip*nmol[idx]*(pops[lau, idx])*aeinst
    alpha_precalc = 1./doppb[idx]*hpip*nmol[idx]*((pops[lal,idx])*beinstl - (pops[lau,idx])*beinstu)

    for iline in range(nline):
        jbar_temp = 0.
        for iphot in range(ds.shape[0]):
            jnu = jnu_dust[iline] + vfac[iphot]*jnu_precalc[iline]
            alpha = alpha_dust[iline] + vfac[iphot]*alpha_precalc[iline]

            if (np.abs(alpha) < eps):
                snu = 0.
            else:
                snu = jnu/alpha/norm[iline]

            tau = alpha*ds[iphot]
            if (tau < negtaulim): # Limit negative opacity
                tau = negtaulim
            
            # Add intensity along line segment
            jbar_temp += vfac[iphot]*(np.exp(-tau)*phot[iline+2, iphot] + (1 - np.exp(-tau))*snu)
        jbar[iline] = jbar_temp


    # Normalize and scale by norm and vsum
    if (vsum > 0.):
        jbar *= norm/vsum # Normalize and scale by norm and vsum

    return jbar
