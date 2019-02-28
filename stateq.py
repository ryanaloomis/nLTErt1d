import numpy as np
from common import eps, hpip, negtaulim
from time import time
from numba import jit, prange


@jit(nopython=True, parallel=True)
def getjbar(ds, vfac, vsum, phot, nmol, doppb, lau, lal, aeinst, beinstu, beinstl, blending, blends, nline, pops, jnu_dust, alpha_dust, norm, idx):
    """
    Set the jbar value for the given radial index.

    Args:
        sim (simulation instance):
        idx (int): Ind
        debug (optional[bool]): If True, print debug messages.

    Returns:
        jbar
    """
    jbar = np.zeros(nline)

    jnu_precalc = 1./doppb[idx]*hpip*nmol[idx]*(pops[lau, idx])*aeinst
    alpha_precalc = 1./doppb[idx]*hpip*nmol[idx]*((pops[lal, idx])*beinstl - (pops[lau,idx])*beinstu)

    for iline in prange(nline):
        jbar_temp = 0
        for iphot in prange(ds.shape[0]):
            jnu = jnu_dust[iline] + vfac[iphot]*jnu_precalc[iline]
            alpha = alpha_dust[iline] + vfac[iphot]*alpha_precalc[iline]

            if (np.abs(alpha) < eps):
                snu = 0.
            else:
                snu = jnu/alpha/norm[iline]

            tau = alpha*ds[iphot]
            if (tau < negtaulim): # Limit negative opacity
                tau = negtaulim

            # Add intensity along line segment
            jbar_temp += vfac[iphot]*(np.exp(-tau)*phot[iline+2, iphot] + (1 - np.exp(-tau))*snu)
        jbar[iline] = jbar_temp

    
    # Line blending contribution
    if blending:
        for iblend in range(blends.shape[0]):
            bjnu = 0.
            balpha = 0.
            iline = int(blends[iblend,0])
            jline = int(blends[iblend,1])
            for iphot in range(ds.shape[0]):
                bjnu = vfac[iphot]*jnu_precalc[jline]
                balpha = vfac[iphot]*alpha_precalc[jline]

                if (np.abs(balpha) < eps):
                    bsnu = 0.
                else:
                    bsnu = bjnu/balpha/norm[jline]

                btau = balpha*ds[iphot]
                if (btau < negtaulim): # Limit negative opacity
                    btau = negtaulim

                # Add intensity along line segment
                jbar[iline] += vfac[iphot]*(np.exp(-btau)*phot[iline+2, iphot] + (1 - np.exp(-btau))*bsnu)
    

    # Normalize and scale by norm and vsum
    if (vsum > 0.):
        jbar *= norm/vsum

    return jbar


@jit(nopython=True, parallel=True)
def getmatrix(part2id, phot, nh2, ne, lau, lal, lcu, lcl, lcu2, lcl2, down, up,
              down2, up2, aeinst, beinstu, beinstl, nline, nlev, ntrans, ntrans2, jbar, idx):
    """
    Calculate the matrix.

    Args:
        LOADS

    Returns:
        mtrx (ndarray): Some matrix.
    """

    # Define the empty arrays first.
    matrix = np.zeros((nlev+1, nlev+1))
    colli = np.zeros((nlev, nlev))
    colli2 = np.zeros((nlev, nlev))

    for t in range(nline):
        k = lau[t]
        l = lal[t]
        matrix[k, k] += beinstu[t]*jbar[t] + aeinst[t]
        matrix[l, l] += beinstl[t]*jbar[t]
        matrix[k, l] += -beinstl[t]*jbar[t]
        matrix[l, k] += -beinstu[t]*jbar[t] - aeinst[t]

    # create collision rate matrix and sum by rows. Add the conservation
    # equation to the left hand side (at position nlev+1).
    for i in range(ntrans):
        colli[lcu[i], lcl[i]] = down[i, idx]
        colli[lcl[i], lcu[i]] = up[i, idx]
    ctot = np.sum(colli, axis=1)

    if part2id:
        for i in range(ntrans2):
            colli2[lcu2[i], lcl2[i]] = down2[i, idx]
            colli2[lcl2[i], lcu2[i]] = up2[i, idx]
        ctot2 = np.sum(colli2, axis=1)

    # Fill the rate matrix with the collisional contributions.
    # check if two second collisional partner exists in model
    if part2id:
        for i in range(nlev):
            matrix[i, i] += nh2[idx]*ctot[i] + ne[idx]*ctot2[i]
        matrix[:-1, :-1] += -nh2[idx]*colli.T - ne[idx]*colli2.T
        matrix[nlev, :-1] = 1.
        matrix[:-1, nlev] = 0.
    else:
        for i in range(nlev):
            matrix[i, i] += nh2[idx]*ctot[i]
        matrix[:-1, :-1] += -nh2[idx]*colli.T
        matrix[nlev, :-1] = 1.
        matrix[:-1, nlev] = 0.

    return matrix



@jit(nopython=True, parallel=True)
def solvematrix(ratem, newpop):
    result = np.linalg.lstsq(ratem, newpop)
    return result[0]



@jit(nopython=True)
def stateq(part2id, phot, nmol, nh2, ne, doppb, lau, lal, lcu, lcl, lcu2, lcl2, down, up, down2, up2, aeinst, beinstu, beinstl, blending, blends, nline, nlev, ntrans, ntrans2, pops, dust, knu, norm, minpop, idx, miniter=10, maxiter=100, tol=1e0-6):
    """
    Iterate for convergence between radiation field and excitation.

    Args:
        sim (simulation instance): Instance of the simulation.
        idx (int): Radial index.
        miniter (optional[int]): Minimum number of iterations.
        maxiter (optional[int]): Maximum number of iterations.
        tol (optional[float]): Tolerate of convergence.

    Returns:
        stater (float): Difference between the values.
    """

    opop = np.zeros(nlev)
    oopop = np.zeros(nlev)

    jnu_dust = dust[:, idx]*knu[:, idx]
    alpha_dust = knu[:, idx]

    ds = phot[0]
    vfac = phot[1]
    vsum = np.sum(vfac)

    # Get updated jbar.
    for iter in range(maxiter):
        jbar = getjbar(ds, vfac, vsum, phot, nmol, doppb, lau, lal, aeinst, beinstu, beinstl, blending, blends, nline, pops, jnu_dust, alpha_dust, norm, idx)

        newpop = np.zeros(nlev + 1)
        newpop[-1] = 1.

        # fill collision rate matrix
        ratem = getmatrix(part2id, phot, nh2, ne, lau, lal, lcu, lcl, lcu2, lcl2, down, up, down2, up2, aeinst, beinstu, beinstl, nline, nlev, ntrans, ntrans2, jbar, idx)

        newpop = solvematrix(ratem, newpop)

        diff = 0
        newpop = np.maximum(newpop, eps)
        oopop = opop
        opop = pops[:, idx]
        pops[:, idx] = newpop[:-1]
        if np.any((np.minimum(newpop[:-1], oopop) > minpop)):
            _diff = np.maximum(np.abs(newpop[:-1] - opop) / newpop[:-1],
                               np.abs(newpop[:-1] - oopop) / newpop[:-1])
            diff = np.max(np.maximum(_diff, diff))

        if (iter > miniter) and (diff < tol):
            continue

    # If not all converged, save diff in staterr
    if diff > tol:
        staterr = diff
    return staterr, pops
