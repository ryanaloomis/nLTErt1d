import numpy as np
from simulation import *
from model import *
from common import *
from vfunc_1d import *
from numerical import *
from time import time

def photon(sim, idx, debug):
    sim.phot *= 0.
    for iphot in range(sim.nphot[idx]):
        #start = time()
        sim.tau *= 0. 
        posn = idx
        firststep = True

        # Assign random position within cell id, direction and velocity, and
        # determine distance ds to edge of cell. Choose position so that the
        # cell volume is equally sampled in volume, the direction so that
        # all solid angles are equally sampled, and the velocity offset
        # homogeneously distributed over +/-2.15 * local Doppler b from
        # local velocity.

        #dummy = ran1()
        dummy = np.random.random()        
        if (sim.model.grid['ra'][idx] > 0.):
            rpos = sim.model.grid['ra'][idx]*(1. + dummy*((sim.model.grid['rb'][idx]/sim.model.grid['ra'][idx])**3 - 1.))**(1./3.)
        else:
            rpos = sim.model.grid['rb'][idx]*dummy**(1./3.)

        #dummy = 2.*ran1() - 1.
        dummy = 2.*np.random.random() - 1.
        phi = np.arcsin(dummy) + np.pi/2.

        if debug: print('[debug] calling velo, iphot = ' + str(iphot))
        vel = sim.model.velo(idx, rpos)
        deltav = (np.random.random() - 0.5)*4.3*sim.model.grid['doppb'][idx] + np.cos(phi)*vel[0]

        # Propagate to edge of cloud by moving from cell edge to cell edge.
        # After the first step (i.e., after moving to the edge of cell id),
        # store ds and vfac in phot(1,iphot) and phot(2,iphot). After
        # leaving the source, add CMB and store intensities in 
        # phot(iline+2,iphot)        

        in_cloud = True

        while in_cloud:
            #t0 = time()
            cosphi = np.cos(phi)
            sinphi = np.sin(phi)

            # Find distance to nearest cell edge
            if (np.abs(phi) < np.pi/2.) or (posn == 0):
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
            #t1 = time()
            #print (t1-t0)*1000.

            # Find "vfac", the velocity line profile factor
            # Number of splines nspline=los_delta_v/local_line_width
            # Number of averaging steps naver=local_delta_v/local_line_width
            if (sim.model.grid['nmol'][posn] > eps):
                #t0 = time()
                b = sim.model.grid['doppb'][posn]
                v1 = vfunc(sim, 0., idx, rpos, phi, deltav)
                v2 = vfunc(sim, ds, idx, rpos, phi, deltav)   
                nspline = np.maximum(1, int(np.abs(v1 - v2)/b))
                vfac = 0.
                for ispline in range(nspline):
                    s1 = ds*(ispline)/nspline
                    s2 = ds*(ispline+1.)/nspline
                    v1 = vfunc(sim, s1, idx, rpos, phi, deltav)
                    v2 = vfunc(sim, s2, idx, rpos, phi, deltav)
                    naver = np.maximum(1, int(np.abs(v1 - v2)/b))
                    for iaver in range(naver):
                        s = s1 + (s2-s1)*(iaver + 0.5)/naver
                        v = vfunc(sim, s, idx, rpos, phi, deltav)
                        vfacsub = np.exp(-(v/b)**2)
                        vfac += vfacsub/naver
                vfac /= nspline
                #t1 = time()

                # backwards integrate dI/ds      
                jnu = sim.dust[:,posn]*sim.knu[:,posn] + vfac*hpip/b*sim.model.grid['nmol'][posn]*sim.pops[sim.mol.lau,posn]*sim.mol.aeinst
                alpha = sim.knu[:,posn] + vfac*hpip/b*sim.model.grid['nmol'][posn]*(sim.pops[sim.mol.lal,posn]*sim.mol.beinstl - sim.pops[sim.mol.lau,posn]*sim.mol.beinstu)
    
                snu = jnu/alpha/sim.norm
                snu[np.abs(alpha) < eps] = 0.

                dtau = alpha*ds
                dtau[dtau < negtaulim] = negtaulim

                if not firststep:
                    sim.phot[2:, iphot] += np.exp(-sim.tau)*(1. - np.exp(-dtau))*snu
                    sim.tau += dtau
                    sim.tau[sim.tau < negtaulim] = negtaulim

                if firststep:
                    sim.phot[0, iphot] = ds
                    sim.phot[1, iphot] = vfac
                    firststep = False
                #t2 = time()
                #print (t1-t0)*1000., (t2-t1)*1000.


            # Update photon position, direction; check if escaped
            posn = posn + dpos
            if (posn >= sim.ncell):
                break        # reached edge of cloud, break

            psi = np.arctan2(ds*sinphi, rpos + ds*cosphi)
            phif = phi - psi
            phi = np.mod(phif, np.pi)
            rpos = rnext

        # Finally, add cmb to memorized i0 incident on cell id
        if (sim.model.tcmb > 0.):
            for iline in range(sim.nline):
                sim.phot[iline+2, iphot] += np.exp(-sim.tau[iline])*sim.cmb[iline]
        #end = time()
        #print end-start

    return





