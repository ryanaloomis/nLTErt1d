import numpy as np
import scipy.constants as sc
from common import eps


def planck(freq, temp):
    """
    Planck function for a given frequency [Hz] and temperature [K].
    TODO: Check output units!

    Args:
        freq (float): Frequency in [Hz].
        temp (float): Temperature in [K].

    Returns:
        Bnu (float): Spectral radiance in [W/m^2/sr^1/Hz]. If the temperature
            is too low (T < eps), returns zero.
    """

    # For low temperature, return zero.
    if temp < eps:
        return 0.0

    Bnu = 2. * sc.h * freq**3 * sc.c**-2
    Bnu /= np.exp(sc.h * freq / (sc.k * temp)) - 1.0

    '''
    I don't think we should need to differentiate. Just use full Planck.
    # Explicitely take Wien approximation for h*ku>>k*T:
    Bnu = 2.*hplanck*((freq/clight)**2.)*freq
    if (hplanck*freq > 100.*kboltz*t):
        Bnu *= np.exp(-hplanck*freq/(kboltz*t))
    else:
        Bnu /= np.exp(hplanck*freq/(kboltz*t))-1.
    '''
    return Bnu


# Do we still need this stuff?

rand_1971 = np.loadtxt("rand_1971.txt")


def ran1(reset=False):
    if reset:
        ran1.counter = 0
    else:
        ran1.counter += 1
        return rand_1971[ran1.counter-1]


ran1.counter = 0
