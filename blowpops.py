from __future__ import print_function
import numpy as np
import scipy.constants as sc


def blowpops(outfile, sim, snr, percent):
    """
    Writes populations of all levels in all grid points to file.

    Args:
        outfile (str): Path of file to write to.
        sim (model instance): Simulation instance.
        snr (float): Minimum signal-to-noise achieved.
        percent (float): Convergence percentage.
    """

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
    popfile.write("gas:dust = " + str(sim.model.g2d) + "\n")
    popfile.write("columns = id,ra,rb,nh,tk,nm,vr,db,td,lp\n")
    popfile.write("molfile = " + str(sim.molfile) + "\n")
    #popfile.write("velo = " + sim.velocity + "\n")
    if sim.kappa_params:
        popfile.write("kappa = " + sim.kappa_params + "\n")
    else:
        popfile.write("kappa = None\n")
    popfile.write("@\n")

    # Write grid, reverting to 'natural' units.
    for idx in range(sim.ncell):

        # I _think_ this is what this parameter is.
        v_turb = 2. * sc.k * sim.model.tkin[idx]
        v_turb /= sim.mol.molweight * sc.m_p
        v_turb = np.sqrt(sim.model.doppb[idx]**2 - v_turb) / 1.0e3

        print_vals = [idx+1,                                # Cell number.
                      sim.model.ra[idx],            # ?
                      sim.model.rb[idx],            # ?
                      sim.model.nh2[idx] / 1.0e6,   # Number density.
                      sim.model.tkin[idx],          # Temperature.
                      sim.model.nmol[idx] / 1.0e6,  # Column density.
                      sim.model.velocities[0,idx] / 1.0e3,    # Velocity.
                      v_turb,                               # Turbulence.
                      sim.model.tdust[idx]]         # Dust temp.
        print_vals.extend(sim.pops[:, idx])
        printstr = ' '.join(['{0:.5e}'.format(val) for val in print_vals])
        print(printstr, file=popfile)
    popfile.close()

    return
