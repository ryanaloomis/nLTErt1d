import numpy as np
from simulation import simulation
from model import model
from common import *
from vfunc_1d import *

def photon(sim, idx, debug):
    phot = np.zeros((sim.nline+2, sim.nphot[idx]))
    tau = np.zeros(sim.nline)

    for iphot in range(sim.nphot[idx])
        posn = idx
        firststep = True

        # Assign random position within cell id, direction and velocity, and
        # determine distance ds to edge of cell. Choose position so that the
        # cell volume is equally sampled in volume, the direction so that
        # all solid angles are equally sampled, and the velocity offset
        # homogeneously distributed over +/-2.15 * local Doppler b from
        # local velocity.

        np.random.seed(sim.seed)
        dummy = np.random.random()
        if (sim.model.grid['ra'][idx] > 0.):
            rpos = sim.model.grid['ra'][idx]*(1. + dummy*((sim.model.grid['rb'][idx]/sim.model.grid['ra'][idx])**3 - 1.))**(1./3.)
        else:
            rpos = sim.model.grid['rb'][idx]*dummy**(1./3.)

        np.random.seed(sim.seed)
        dummy = 2.*np.random.random() - 1.
        phi = np.sin(dummy) + np.pi/2.

        if debug: print('[debug] calling velo, iphot = ' + str(iphot))
        vel = sim.model.velo(idx, rpos)
        np.random.seed(sim.seed)
        deltav = (np.random.random() - 0.5)*4.3*sim.model.grid['doppb'][idx] + np.cos(phi)*vel[0]


        # Propagate to edge of cloud by moving from cell edge to cell edge.
        # After the first step (i.e., after moving to the edge of cell id),
        # store ds and vfac in phot(1,iphot) and phot(2,iphot). After
        # leaving the source, add CMB and store intensities in 
        # phot(iline+2,iphot)        

        in_cloud = True:

        while in_cloud:
            cosphi = np.cos(phi)
            sinphi = np.sin(phi)

            # Find distance to nearest cell edge
            if (np.abs(phi) < np.pi/2.) or (posn == 1):
                dpos = 1
                rnext = sim.model.grid['rb'][posn]*(1 + delta)
                bac = 4.*((rpos*cosphi)**2. - rpos**2. + rnext**2.)
            else:
                dpos = -1
                rnext = sim.model.grid['ra'][posn]*(1 - delta)
                bac = 4.*((rpos*cosphi)**2. - rpos**2. + rnext**2.)
                if (bac < 0.):
                    dpos = 1
                    rnext = sim.model.grid['rb'][posn]*(1 + delta)
                    bac = 4.*((rpos*cosphi)**2. - rpos**2. + rnext**2.)

            dsplus = -0.5*(2.*rpos*cosphi + np.sqrt(bac))
            dsminn = -0.5*(2.*rpos*cosphi - np.sqrt(bac))

            if (dsminn*dsplus == 0.):
                ds = 0.
            else:
                if (dsplus < 0.): ds = dsminn
                if (dsminn < 0.): ds = dsplus
                if (dsminn*dsplus > 0.): ds = np.min([dsplus, dsminn])


            # Find "vfac", the velocity line profile factor
            # Number of splines nspline=los_delta_v/local_line_width
            # Number of averaging steps naver=local_delta_v/local_line_width

            if (nmol[posn] > eps):
                b = sim.model.grid['doppb'][posn]
                v1 = vfunc(0., idx, rpos, phi)
                v2 = vfunc(ds, idx, rpos, phi)
                nspline = np.max(1, int(np.abs(v1 - v2)/b))
                vfac = 0.
                for ispline in range(nspline):
                    s1 = ds*(ispline)/nspline
                    s2 = ds*(ispline+1.)/nspline
                    v1 = vfunc(s1, idx, rpos, phi)
                    v2 = vfunc(s2, idx, rpos, phi)
                    naver = np.max(1, int(np.abs(v1 - v2)/b))
                    for iaver in range(naver):
                        s = s1 + (s2-s1)*(iaver + 0.5)/naver
                        v = vfunc(s, idx, rpos, phi)
                        vfacsub = np.exp(-(v/b)**2)
                        vfac = vfacsub/naver
                vfac /= nspline

                # backwards integrate dI/ds            
                for iline in range(sim.nline):
                    jnu = sim.model.dust[iline,posn]*sim.model.knu[iline,posn] + vfac*hpip/b*sim.model.grid['nmol'][posn]*sim.pops[sim.mol.lau[iline],posn]*sim.mol.aeinst[iline] # TODO

                    alpha = self.model.knu[iline,posn] + vfac*hpip/b*sim.model.grid['nmol'][posn]*(sim.pops[sim.mol.lal[iline],posn]*sim.mol.beinstl[iline] - sim.pops[lau[iline],posn]*sim.mol.beinstu[iline]) # TODO

                    if (np.abs(alpha) < eps):
                        snu = 0.
                    else:
                        snu = jnu/alpha/norm[iline] # TODO

                    dtau = alpha*ds
                    if (dtau < negtaulim): # Limit negative opacity
                        dtau = negtaulim

                    if not firststep:
                        phot[iline+2, iphot] += np.exp(-tau[iline])*(1. - np.exp(-dtau))*snu
                        tau[iline] = tau[iline] + dtau
                        if (tau[iline] < negtaulim):
                            tau[iline] = negtaulim

                if firststep:
                    phot[0, iphot] = ds
                    phot[1, iphot] = vfac
                    firststep = False


            # Update photon position, direction; check if escaped
            posn = posn + dpos
            if (posn > ncell):
                continue        # reached edge of cloud, break

            psi = np.arctan2(ds*sinphi, rpos + ds*cosphi)
            phif = phi - psi
            phi = np.mod(phif, np.pi)
            rpos = rnext

        # Finally, add cmb to memorized i0 incident on cell id
        if (sim.model.tcmb > 0.):
            for iline in range(nline):
                phot[iline+2, iphot] += np.exp(-tau[iline])*sim.model.cmb[iline] # TODO

    return





