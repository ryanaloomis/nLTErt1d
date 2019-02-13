import numpy as np
from simulation import *
from model import *
from common import *
from numba import jit, prange


@jit(nopython=True, parallel=True)
def vfunc(v, s, rpos, phi, vphot):
    # Get direction and position at location s along l.o.s.
    psis = np.arctan2(s*np.sin(phi), rpos + s*np.cos(phi))
    phis = phi - psis
    r = np.sqrt(rpos**2. + s**2. + 2.*rpos*s*np.cos(phi))

    # vfunc is velocity difference between between photon and gas
    # projected on l.o.s.
    vfunc = vphot - np.cos(phis)*v[0]
    return vfunc


@jit(nopython=True)
def photon(fixseed, stage, ra, rb, nmol, doppb, vel_grid, lau, lal, aeinst, beinstu, beinstl, tcmb, ncell, nline, pops, dust, knu, norm, cmb, nphot, idx):
    phot = np.zeros((nline+2, nphot))
    if stage ==1:
        np.random.seed(fixseed)

    for iphot in range(nphot):
        tau = np.zeros(nline)
        posn = idx
        firststep = True

        # Assign random position within cell id, direction and velocity, and
        # determine distance ds to edge of cell. Choose position so that the
        # cell volume is equally sampled in volume, the direction so that
        # all solid angles are equally sampled, and the velocity offset
        # homogeneously distributed over +/-2.15 * local Doppler b from
        # local velocity.

        dummy = np.random.random()
        if (ra[idx] > 0.):
            rpos = ra[idx]*(1. + dummy*((rb[idx]/ra[idx])**3 - 1.))**(1./3.)
        else:
            rpos = rb[idx]*dummy**(1./3.)

        dummy = 2.*np.random.random() - 1.
        phi = np.arcsin(dummy) + np.pi/2.

        vel = vel_grid[idx]
        deltav = (np.random.random() - 0.5)*4.3*doppb[idx] + np.cos(phi)*vel[0]

        # Propagate to edge of cloud by moving from cell edge to cell edge.
        # After the first step (i.e., after moving to the edge of cell id),
        # store ds and vfac in phot(1,iphot) and phot(2,iphot). After
        # leaving the source, add CMB and store intensities in
        # phot(iline+2,iphot)

        in_cloud = True

        while in_cloud:
            cosphi = np.cos(phi)
            sinphi = np.sin(phi)

            # Find distance to nearest cell edge
            if (np.abs(phi) < np.pi/2.) or (posn == 0):
                dpos = 1
                rnext = rb[posn]*(1 + delta)
                bac = 4.*((rpos*cosphi)**2. - rpos**2. + rnext**2.)
            else:
                dpos = -1
                rnext = ra[posn]*(1 - delta)
                bac = 4.*((rpos*cosphi)**2. - rpos**2. + rnext**2.)
                if (bac < 0.):
                    dpos = 1
                    rnext = rb[posn]*(1 + delta)
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
                b = doppb[posn]
                v1 = vfunc(vel_grid[idx], 0., rpos, phi, deltav)
                v2 = vfunc(vel_grid[idx], ds, rpos, phi, deltav)
                nspline = np.maximum(1, int(np.abs(v1 - v2)/b))
                vfac = 0.
                for ispline in range(nspline):
                    s1 = ds*(ispline)/nspline
                    s2 = ds*(ispline+1.)/nspline
                    v1 = vfunc(vel_grid[idx], s1, rpos, phi, deltav)
                    v2 = vfunc(vel_grid[idx], s2, rpos, phi, deltav)
                    naver = np.maximum(1, int(np.abs(v1 - v2)/b))
                    for iaver in range(naver):
                        s = s1 + (s2-s1)*(iaver + 0.5)/naver
                        v = vfunc(vel_grid[idx], s, rpos, phi, deltav)
                        vfacsub = np.exp(-(v/b)**2)
                        vfac += vfacsub/naver
                vfac /= nspline

                # backwards integrate dI/ds
                jnu = dust[:,posn]*knu[:,posn] + vfac*hpip/b*nmol[posn]*pops[lau,posn]*aeinst
                alpha = knu[:,posn] + vfac*hpip/b*nmol[posn]*(pops[lal,posn]*beinstl - pops[lau,posn]*beinstu)

                snu = jnu/alpha/norm
                snu[np.abs(alpha) < eps] = 0.

                dtau = alpha*ds
                dtau[dtau < negtaulim] = negtaulim

                if not firststep:
                    phot[2:, iphot] += np.exp(-tau)*(1. - np.exp(-dtau))*snu
                    tau += dtau
                    tau[tau < negtaulim] = negtaulim

                if firststep:
                    phot[0, iphot] = ds
                    phot[1, iphot] = vfac
                    firststep = False

            # Update photon position, direction; check if escaped
            posn = posn + dpos
            if (posn >= ncell):
                break        # reached edge of cloud, break

            psi = np.arctan2(ds*sinphi, rpos + ds*cosphi)
            phif = phi - psi
            phi = np.mod(phif, np.pi)
            rpos = rnext

        # Finally, add cmb to memorized i0 incident on cell id
        if (tcmb > 0.):
            for iline in range(nline):
                phot[iline+2, iphot] += np.exp(-tau[iline])*cmb[iline]

    return phot
