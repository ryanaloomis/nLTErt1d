import numpy as np
from common import *
from simulation import *
from scipy.interpolate import CubicSpline

# Returns dust emissivity in m2/kg_dust at frequency freq and in cell id.
# Ossenkopf & Henning Jena models.
#
# Useage in amc.inp:  kappa=jena,TYPE,COAG
# where TYPE = bare / thin / thick
# and   COAG = no / e5 / e6 / e7 / e8

def generate_kappa(kappa_params):
    if kappa_params:
        params = kappa_params.split(',')

    else:
        # default case, set kappa to 0
        def kappa(idx, freq):
            return 0.
        return kappa
    

    if params[0] == 'jena':
        filename = "kappa/jena_" + params[1] + "_" + params[2] + ".tab"
        table = np.loadtxt(filename)
        lamtab = table[:,0]/1.e6
        kaptab = table[:,1]

        interp_func = CubicSpline(np.log10(lamtab), np.log10(kaptab), extrapolate=True)

        def kappa(idx, freq):
            lam_lookup = clight/freq
            kap_interp = 10**interp_func(np.log10(lam_lookup))
            # ...in m2/kg_dust: (0.1 converts cm2/g_dust to m2/kg_dust)
            return kap_interp*0.1
        return kappa

    elif params[0] == 'powerlaw':
        freq0 = float(params[1])
        kappa0 = float(params[2])
        beta = float(params[3])

        def kappa(idx, freq):
            # Simple power law behavior; kappa=powerlaw,freq0,kappa0,beta where freq0 is in Hz, kappa0 in cm2/g_dust, and beta is freq.index.
            # ...in m2/kg_dust: (0.1 converts cm2/g to m2/kg)
            return 0.1*kappa0*(freq/freq0)**beta
        return kappa

    else:
        raise Exception("ERROR: Please supply valid kappa parameters.")
