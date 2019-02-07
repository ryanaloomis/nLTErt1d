from __future__ import print_function
import numpy as np
from common import *

#     Output routine of AMC. Writes populations of all levels in all
#     grid points to file.


def blowpops(outfile, sim, snr, percent):
    # Open file (overwrites old file)
    popfile = open(outfile, 'w')

    # Write out the header
    popfile.write("#AMC: version 0.1.0 / ral / feb2019\n")
    popfile.write("#AMC: output file\n")
    popfile.write("#AMC: fixset convergence limit = " + str(sim.fixset) + "\n")
    popfile.write("#AMC: convergence reached = " + str(1./snr) + "\n")
    popfile.write("#AMC: requested snr = " + str(sim.goalsnr) + "\n")
    popfile.write("#AMC: minimum snr = " + str(snr) + "\n")
    popfile.write("#AMC: " + str(percent) + "% converged\n")
    popfile.write("rmax = " + str(sim.model.rmax) + "\n")
    popfile.write("ncell = " + str(sim.model.ncell) + "\n")
    popfile.write("tcmb = " + str(sim.model.tcmb) + "\n")
    popfile.write("gas:dust = " + str(sim.model.gas2dust) + "\n")
    popfile.write("columns = id,ra,rb,nh,tk,nm,vr,db,td,lp\n")
    popfile.write("molfile = " + str(sim.molfile) + "\n")
    popfile.write("velo = " + sim.velocity + "\n")
    popfile.write("kappa = " + sim.kappa_params + "\n")
    popfile.write("@\n")

    # Write grid, reverting to 'natural' units
    for idx in range(sim.ncell):
        print_vals = [idx+1, sim.model.grid["ra"][idx], sim.model.grid["rb"][idx], sim.model.grid["nh2"][idx]/1.e6, sim.model.grid["tkin"][idx], sim.model.grid["nmol"][idx]/1.e6, sim.model.grid["vr"][idx]/1.e3, np.sqrt((sim.model.grid["doppb"][idx])**2-2.*kboltz/(sim.mol.molweight*amu)*sim.model.grid["tkin"][idx])/1.e3, sim.model.grid["tdust"][idx]]
        print_vals.extend(sim.pops[:,idx])
        printstr = ' '.join(['{0:.5e}'.format(val) for val in print_vals])
        print(printstr, file=popfile)
    popfile.close()

    return
