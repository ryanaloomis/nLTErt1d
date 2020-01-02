# Stuff to get around the different folders for now.
import matplotlib.pylab as pl
import numpy as np
from time import time

import sys
sys.path.append('../src/')
from simulation import simulation

if __name__ == "__main__":

    t0 = time()

    # Set up simulation
    sim = simulation(source='example.mdl',
                     outfile='example.pop',
                     # molfile='hco+.dat',
                     molfile='cn-hfs.dat',
                     goalsnr=2,
                     nphot=100,
                     kappa='jena,thin,e5',
                     blending=False,
                     )

    # Calculate the level populations
    sim.calc_pops()

    # Define the lines to observe.
    lines = [4, 5, 10]

    # Do the raytracing
    velax, intens, tau = sim.raytrace(v_min=-3e3,
                                      v_max=3e3,
                                      dv=250.0,
                                      rt_lines=lines)
    # Plot the intensities.
    for idx, line in enumerate(lines):
        freqax = sim.velax_to_freqax(velax, line_idx=line)
        pl.step(freqax, intens[idx], where='mid')
        pl.show()

    # # Plot the optical depths.
    # for idx, line in enumerate(lines):
    #     pl.plot(tau[idx])
    #     pl.show()

    np.save("output.npy", intens)

    print('Took in total {:.1f} seconds.'.format(time()-t0))
