import numpy as np
from common import *

#     Returns the value of the planck function at a given frequency, freq, and temperature, t.
def planck(freq, t):

    if (t < eps):
        planck=0.

    else:
    #     Explicitely take Wien approximation for h*nu>>k*T:
        if (hplanck*freq > 100.*kboltz*t):
            planck = 2.*hplanck*((freq/clight)**2.)*freq*np.exp(-hplanck*freq/(kboltz*t))
        else:
            planck = 2.*hplanck*((freq/clight)**2.)*freq/(np.exp(hplanck*freq/(kboltz*t))-1.)

    return planck


rand_1971 = np.loadtxt("rand_1971.txt")

def ran1(reset=False):
    if reset:
        ran1.counter = 0
    else:
        ran1.counter += 1
        return rand_1971[ran1.counter-1]
    
ran1.counter=0
