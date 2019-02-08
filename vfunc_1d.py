import numpy as np
from simulation import *
from model import *
from common import *

def vfunc(sim, s, idx, rpos, phi, vphot):
    # Get direction and position at location s along l.o.s. 
    psis = np.arctan2(s*np.sin(phi), rpos + s*np.cos(phi))
    phis = phi - psis
    r = np.sqrt(rpos**2. + s**2. + 2.*rpos*s*np.cos(phi))

    # Get velocity vector of the gas at this position
    v = sim.model.velo(idx, r)

    # vfunc is velocity difference between between photon and gas
    # projected on l.o.s.
    vfunc = vphot - np.cos(phis)*v[0]

    return vfunc
