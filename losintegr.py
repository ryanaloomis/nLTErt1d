import numpy as np
from simulation import *
from model import *
from common import *
from numba import jit, prange

#@jit(nopython=True, parallel=True)
def vfunc(v, s, rpos, phi, vphot):
    # Get direction and position at location s along l.o.s.
    psis = np.arctan2(s*np.sin(phi), rpos + s*np.cos(phi))
    phis = phi - psis
    r = np.sqrt(rpos**2. + s**2. + 2.*rpos*s*np.cos(phi))

    # vfunc is velocity difference between between photon and gas
    # projected on l.o.s.
    vfunc = vphot - np.cos(phis)*v[0]
    return vfunc


#@jit(nopython=True, fastmath=True, parallel=True)
def calcLineAmp(b, vel_grid, rpos, ds, phi, deltav, idx):
    v1 = vfunc(vel_grid[:, idx], 0., rpos, phi, deltav)
    v2 = vfunc(vel_grid[:, idx], ds, rpos, phi, deltav)
    nspline = np.maximum(1, int(np.abs(v1 - v2)/b))
    vfac = 0.
    for ispline in range(nspline):
        s1 = ds*(ispline)/nspline
        s2 = ds*(ispline+1.)/nspline
        v1 = vfunc(vel_grid[:, idx], s1, rpos, phi, deltav)
        v2 = vfunc(vel_grid[:, idx], s2, rpos, phi, deltav)
        naver = np.maximum(1, int(np.abs(v1 - v2)/b))
        for iaver in range(naver):
            s = s1 + (s2-s1)*(iaver + 0.5)/naver
            v = vfunc(vel_grid[:, idx], s, rpos, phi, deltav)
            vfacsub = np.exp(-(v/b)**2)
            vfac += vfacsub/naver
    vfac /= nspline
    return vfac


def losintegr(rmax, ra, rb, nmol, nh2, doppb, vel_grid, lau, lal, aeinst, beinstu, beinstl, blending, blends, tcmb, ncell, pops, dust, knu, norm, cmb, nchan, vcen, velres, rt_lines):
    reflect=True
    tau = np.zeros((len(rt_lines), nchan))
    intens = np.zeros((len(rt_lines), nchan))

    # Raytrace photons along cells, with photon velocity determined by
    # channel step size. Start at front of source

    rpos = rmax*(1. - delta)
    phi = -np.pi

    posn = ncell-1
    
    costheta=1.
    sintheta=0.
    cosphi=np.cos(phi)
    sinphi=np.sin(phi)


    # Propagate to edge of cloud by moving from cell edge to cell edge.
    # After leaving the source, add CMB and store intensities

    in_cloud = True

    while in_cloud:
        # Find distance to nearest cell edge
        print(posn)
        dpos = -1
        rnext = ra[posn]*(1 - delta)
        bac = 4.*((rpos*cosphi)**2. - rpos**2. + rnext**2.)

        dsplus = -0.5*(2.*rpos*cosphi + np.sqrt(bac))
        dsminn = -0.5*(2.*rpos*cosphi - np.sqrt(bac))

        if (dsminn*dsplus == 0.):
            ds = 0.
        else:
            if (dsplus < 0.): ds = dsminn
            if (dsminn < 0.): ds = dsplus
            if (dsminn*dsplus > 0.): ds = np.min(np.array([dsplus, dsminn]))

        # Find "vfac", the velocity line profile factor
        # Number of splines nspline=los_delta_v/local_line_width
        # Number of averaging steps naver=local_delta_v/local_line_width

        if (nmol[posn] > eps):
            for ichan in range(nchan):
                deltav=(ichan-vcen)*velres
                b = doppb[posn]
                vfac = calcLineAmp(b, vel_grid, rpos, ds, phi, deltav, posn)

                for idx, iline in enumerate(rt_lines):
                    # backwards integrate dI/ds
                    jnu = dust[iline,posn]*knu[iline,posn] + vfac*hpip/b*nmol[posn]*pops[lau,posn][iline]*aeinst[iline]
                    alpha = knu[iline,posn] + vfac*hpip/b*nmol[posn]*(pops[lal,posn][iline]*beinstl[iline] - pops[lau,posn][iline]*beinstu[iline])

                    snu = jnu/alpha/norm[iline]
                    if np.abs(alpha) < 0: snu = 0.

                    dtau = alpha*ds
                    if dtau < negtaulim: dtau = negtaulim  

                    intens[idx,ichan] += np.exp(-tau[idx,ichan])*(1.-np.exp(-dtau))*snu
                    tau[idx,ichan] += dtau
                    
                    # Line blending step
                    if blending:
                        for iblend in range(blends.shape[0]):
                            if iline == int(blends[iblend,0]):
                                bjnu = 0.
                                balpha = 0.
                                jline = int(blends[iblend,1])
                                bdeltav = blends[iblend,2]

                                velproj = deltav - bdeltav
                                bvfac = calcLineAmp(b, vel_grid, rpos, ds, phi, velproj, posn)

                                bjnu = bvfac*hpip/b*nmol[posn]*pops[lau,posn][jline]*aeinst[jline]
                                balpha = bvfac*hpip/b*nmol[posn]*(pops[lal,posn][jline]*beinstl[jline] - pops[lau,posn][jline]*beinstu[jline])

                                if np.abs(balpha) < eps: 
                                    bsnu = 0.
                                else:                            
                                    bsnu = bjnu/balpha/norm[jline]                        

                                bdtau = balpha*ds
                                if bdtau < negtaulim: bdtau = negtaulim                   
                        
                                intens[idx,ichan] += np.exp(-tau[idx,ichan])*(1. - np.exp(-bdtau))*bsnu
                                tau[idx,ichan] += bdtau


        # Update photon position, direction; check if escaped
        posn += dpos
        if (posn < 0 or posn >= ncell):
            break        # reached edge of cloud, break

        rpos = rnext


    # if reflecting, propogate through the whole cloud again in the other direction (but now velocities are reversed)
    if reflect==True:
        rpos = 0. + delta
        phi = np.pi
        posn = 0
        costheta=1.
        sintheta=0.
        cosphi=np.cos(phi)
        sinphi=np.sin(phi)

        while in_cloud:
            # Find distance to nearest cell edge
            print(posn)
            dpos = 1
            rnext = rb[posn]*(1 + delta)
            bac = 4.*((rpos*cosphi)**2. - rpos**2. + rnext**2.)

            dsplus = 0.5*(2.*rpos*cosphi + np.sqrt(bac))
            dsminn = 0.5*(2.*rpos*cosphi - np.sqrt(bac))

            if (dsminn*dsplus == 0.):
                ds = 0.
            else:
                if (dsplus < 0.): ds = dsminn
                if (dsminn < 0.): ds = dsplus
                if (dsminn*dsplus > 0.): ds = np.min(np.array([dsplus, dsminn]))

            # Find "vfac", the velocity line profile factor
            # Number of splines nspline=los_delta_v/local_line_width
            # Number of averaging steps naver=local_delta_v/local_line_width

            if (nmol[posn] > eps):
                for ichan in range(nchan):
                    deltav=(ichan-vcen)*velres
                    b = doppb[posn]
                    vfac = calcLineAmp(b, -vel_grid, rpos, ds, phi, deltav, posn)

                    for idx, iline in enumerate(rt_lines):
                        # backwards integrate dI/ds
                        jnu = dust[iline,posn]*knu[iline,posn] + vfac*hpip/b*nmol[posn]*pops[lau,posn][iline]*aeinst[iline]
                        alpha = knu[iline,posn] + vfac*hpip/b*nmol[posn]*(pops[lal,posn][iline]*beinstl[iline] - pops[lau,posn][iline]*beinstu[iline])

                        snu = jnu/alpha/norm[iline]
                        if np.abs(alpha) < 0: snu = 0.

                        dtau = alpha*ds
                        if dtau < negtaulim: dtau = negtaulim  

                        intens[idx,ichan] += np.exp(-tau[idx,ichan])*(1.-np.exp(-dtau))*snu
                        tau[idx,ichan] += dtau
                        
                        # Line blending step
                        if blending:
                            for iblend in range(blends.shape[0]):
                                if iline == int(blends[iblend,0]):
                                    bjnu = 0.
                                    balpha = 0.
                                    jline = int(blends[iblend,1])
                                    bdeltav = blends[iblend,2]

                                    velproj = deltav - bdeltav
                                    bvfac = calcLineAmp(b, vel_grid, rpos, ds, phi, velproj, posn)

                                    bjnu = bvfac*hpip/b*nmol[posn]*pops[lau,posn][jline]*aeinst[jline]
                                    balpha = bvfac*hpip/b*nmol[posn]*(pops[lal,posn][jline]*beinstl[jline] - pops[lau,posn][jline]*beinstu[jline])

                                    if np.abs(balpha) < eps: 
                                        bsnu = 0.
                                    else:                            
                                        bsnu = bjnu/balpha/norm[jline]                        

                                    bdtau = balpha*ds
                                    if bdtau < negtaulim: bdtau = negtaulim                   
                            
                                    intens[idx,ichan] += np.exp(-tau[idx,ichan])*(1. - np.exp(-bdtau))*bsnu
                                    tau[idx,ichan] += bdtau


            # Update photon position, direction; check if escaped
            posn += dpos
            if (posn < 0 or posn >= ncell):
                break        # reached edge of cloud, break

            rpos = rnext


    # Finally, add cmb to memorized i0 incident on cell id
    if (tcmb > 0.):
        for idx, iline in enumerate(rt_lines):
            intens[idx, :] += np.exp(-tau[idx, :])*cmb[iline]

    return intens, tau
