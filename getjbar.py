from common import eps, negtaulim
import scipy.constants as sc
import numpy as np
# from simulation import *
# from model import *


def getjbar(sim, idx, debug=False):
    """
    Set the jbar value for the given radial index.

    Args:
        sim (simulation instance):
        idx (int): Ind
        debug (optional[bool]): If True, print debug messages.

    Returns:
        None
    """

    # I don't think we need to declare these.
    # sim.mol.jbar = np.zeros(sim.nline)
    # jnu_dust = np.zeros(sim.nline)
    # alpha_dust = np.zeros(sim.nline)
    # vsum = 0.

    alpha_dust = sim.knu[:, idx]
    jnu_dust = sim.dust[:, idx] * alpha_dust

    # Define some variables.
    ds = sim.phot[0]
    vfac = sim.phot[1]
    vsum = np.sum(vfac)
    hpip = sc.h * sc.c * np.pi**(-1.5) / 4.

    # Emission.
    jnu = vfac * hpip * sim.pops[sim.mol.lau, idx] * sim.mol.aeinst
    jnu = jnu[:, np.newaxis] * sim.model.grid['nmol'][idx]
    jnu /= sim.model.grid['doppb'][idx]
    jnu += jnu_dust[:, np.newaxis]

    # Absorption.
    alpha = 0.0

    # jnu = jnu_dust[:, np.newaxis] + vfac[np.newaxis, :] / sim.model.grid['doppb'][idx] * hpip * sim.model.grid['nmol'][idx] * (sim.pops[sim.mol.lau, idx])[:, np.newaxis] * sim.mol.aeinst[:, np.newaxis]
    alpha = alpha_dust[:, np.newaxis] + vfac[np.newaxis, :] / sim.model.grid['doppb'][idx] * hpip * sim.model.grid['nmol'][idx]*((sim.pops[sim.mol.lal, idx])[:, np.newaxis]*sim.mol.beinstl[:, np.newaxis] - (sim.pops[sim.mol.lau, idx])[:, np.newaxis]*sim.mol.beinstu[:, np.newaxis])

    # Source function.
    snu = jnu / alpha / sim.norm[:, np.newaxis]
    snu[np.abs(alpha) < eps] = 0.

    # Optical depth.
    tau = alpha * ds[np.newaxis, :]
    tau[tau < negtaulim] = negtaulim

    sim.mol.jbar = np.exp(-tau) * sim.phot[2:, :] + (1 - np.exp(-tau)) * snu
    sim.mol.jbar = vfac[np.newaxis, :] * ()
    sim.mol.jbar = np.sum(sim.mol.jbar, axis=1)

    # Normalize and scale by norm and vsum
    if (vsum > 0.):
        sim.mol.jbar *= sim.norm / vsum

    return
