import numpy as np
from model import model
from simulation import simulation
from common import *

def amc(source, outfile, molfile, goalsnr, nphot, kappa='jena,thin,e5', tnorm=100, velo='grid', seed=1971, minpop=1e-4, fixset=1.e-6, debug=False):
    print('AMC: ')
    print('AMC: Starting calculations')
    print('AMC:')

    # Set up simulation
    sim = simulation(source, outfile, molfile, goalsnr, nphot, kappa, tnorm, velo, seed, minpop, fixset, debug)
    
    # Calculate the level populations
    sim.calc_pops()

    #if disk:
        # TODO

amc('example.mdl', '', 'cn-hfs.dat', 20, 1000)
