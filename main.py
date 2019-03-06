from simulation import simulation
import matplotlib.pylab as pl
import numpy as np

if __name__ == "__main__":
    # Set up simulation
    sim = simulation('CN.mdl', 'CN.pop', 'cn-hfs.dat', 15, 300, kappa='jena,thin,e5', blending=True, rt_lines=[41], velres=0.02, nchan=300)

    # Calculate the level populations
    sim.calc_pops()

    # Do the raytracing
    intens, tau = sim.raytrace()

    for idx, line in enumerate([41]):
        pl.plot(intens[idx])
        pl.show()

    for idx, line in enumerate([41]):
        pl.plot(tau[idx])
        pl.show()

    np.save("output.npy", intens)
