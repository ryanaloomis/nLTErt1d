import numpy as np
from simulation import simulation
from model import model
from common import *


def vfunc(s, idx, rpos, phi):
    # Get direction and position at location s along l.o.s. 
    psis = np.arctan2(s*np.sin(phi), rpos + s*np.cos(phi))
    phis = phi - psis
    r = np.sqrt(rpos**2. + s**2. + 2.*rpos*s*np.cos(phi))

    # Get velocity vector of the gas at this position
    v = velo(idx, r)

    # vfunc is velocity difference between between photon and gas
    # projected on l.o.s.
    vfunc = vphot - np.cos(phis)*v[0]

    return vfunc
