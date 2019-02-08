"""
Common constants. Can probably move these to class variables.
"""

import numpy as np

eps = 1.e-30        # number of photons.
max_phot = 100000   # max number of photons.
negtaulim = -30.    # negative optical depth limit.
delta = 1.0e-10     # delta?

# Can define these elsewhere.
spi = np.sqrt(np.pi)
clight = 2.997924562e8
hplanck = 6.626196e-34
kboltz = 1.380622e-23
amu = 1.6605402e-27
hpip = hplanck*clight/4./np.pi/spi
