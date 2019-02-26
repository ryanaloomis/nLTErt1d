from simulation import simulation
import matplotlib.pylab as pl

if __name__ == "__main__":
    # Set up simulation
    sim = simulation('example.mdl', 'example.pop', 'hco+.dat', 20, 1000, kappa='jena,thin,e5', blending=False)

    # Calculate the level populations
    sim.calc_pops()

    # Do the raytracing
    intens, tau = sim.raytrace()

    for idx, line in enumerate([0,1,2]):
        pl.plot(intens[idx])
        pl.show()

    for idx, line in enumerate([0,1,2]):
        pl.plot(tau[idx])
        pl.show()
