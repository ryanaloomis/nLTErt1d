import numpy as np

eps = 1.e-30
max_phot = 100000

spi = np.sqrt(np.pi)

clight = 2.997924562e8
hplanck = 6.626196e-34
kboltz = 1.380622e-23
amu = 1.6605402e-27
delta=1.0e-10
hpip = hplanck*clight/4./np.pi/spi


negtaulim = -30.
