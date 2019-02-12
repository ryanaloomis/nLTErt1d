import numpy as np
from model import model
from molecule import molecule
from common import eps, max_phot
from numerical import planck
from kappa import generate_kappa
from photon_1d import photon
from stateq import stateq
from blowpops import blowpops
from time import time
import scipy.constants as sc
from numba import jit


class simulation:
    """The main simulation class."""

    def __init__(self, source, outfile, molfile, goalsnr, nphot, kappa=None,
                 tnorm=2.735, velocity='grid', seed=1971, minpop=1e-4,
                 fixset=1.e-6, debug=False):
        """
        Initlize a simulation.

        Args:
            source (str): Model file.
            outfile (str): File to write population levels to.
            molfile (str): Molecular data file in the LAMDA format.
            goalsnr (float): Goal signal-to-noise ratio of the run.
            nphot (float): Number of photons to use in the radiative transfer.
            kappa (optional[str]): A string decribing the dust parameters. For
                the use of Ossenkopf & Henning (1994) opacities it must take
                the form:

                    kappa_params = 'jena, TYPE, COAG'

                where TYPE must be 'bare', 'thin' or 'thick' and COAG must be
                'no', 'e5', 'e6', 'e7' or 'e8'. Otherwise a power law profile
                can be included. Alternatively, a simple power law can be used
                where the parameters are given by:

                    kappa_params = 'powerlaw, freq0, kappa0, beta'

                where freq0 is in [Hz], kappa0 in [cm^2/g], and beta is the
                frequency index. If nothing is given, we assume no opacity.
            tnorm (optional[float]): Background temperature in [K]. Default is
                the CMB at 2.735K.
            velo (optional[str]): Type of velocity structure to use.
            seed (optional[int]): Seed for the random number generators.
            minpop (optional[float]): Minimum population for each energy level.
            fixset (optional [float]): The smallest number to be counted.
            debug (optional[bool]): Pring debugging messages.
        """

        self.source = source
        self.outfile = outfile
        self.molfile = molfile
        self.goalsnr = goalsnr
        # not setting nphot yet, setting later as array w/ size ncell
        self.kappa_params = kappa
        self.tnorm = tnorm
        self.velocity = velocity
        self.seed = seed
        self.minpop = minpop
        self.fixset = fixset
        self.debug = debug

        t0 = time()
        # Read in the source model
        if self.debug:
            print('[debug] reading in source model')
        self.model = model(self.source, 'ratran', self.debug) # currently setting to ratran as default
        self.ncell = self.model.ncell

        # Check to make sure that there's a valid velocity field
        if self.velocity == 'grid':
            if 'vr' not in self.model.grid:
                raise Exception('Velicity mode specified as grid but no valid '
                                'velocity field included in model.')
        else:
            try:
                # User velocity function override the grid lookup function.
                velo_file = self.velocity
                from velo_file import velo as user_velo
                self.model.velo = user_velo
            except:
                raise Exception('ERROR: Could not read velocity function from'
                                'file ' + velo_file)

        # Read in the molfile
        if self.debug:
            print('[debug] reading molecular data file')
        self.mol = molecule(self, self.molfile, self.debug)
        self.nlev = self.mol.nlev
        self.nline = self.mol.nline
        self.ntrans = self.mol.ntrans
        self.ntrans2 = self.mol.ntrans2
        self.ntemp = self.mol.ntemp
        self.ntemp2 = self.mol.ntemp2

        # Also initialize dust emissivity, converting from m2/kg_dust
        # to "m-1 per H2/cm3" (so that tau_dust=knu)
        self.norm = np.zeros(self.nline)
        self.cmb = np.zeros(self.nline)
        for iline in range(self.nline):
            self.norm[iline] = planck(self.mol.freq[0], tnorm)
            if (self.model.tcmb > 0.):
                self.cmb[iline] = planck(self.mol.freq[iline],
                                         self.model.tcmb) / self.norm[iline]

        # Parse kappa parameters and generate the kappa function
        self.kappa = generate_kappa(self.kappa_params)

        # Do not normalize dust; will be done in photon
        self.knu = np.zeros((self.nline, self.ncell))
        self.dust = np.zeros((self.nline, self.ncell))
        for iline in range(self.nline):
            for idx in range(self.ncell):
                self.knu[iline, idx] = self.kappa(idx, self.mol.freq[iline]) * 2.4 * sc.m_p / self.model.gas2dust*self.model.grid['nh2'][idx]
                self.dust[iline, idx] = planck(self.mol.freq[iline],
                                               self.model.grid['tdust'][idx])

        # Set up the Monte Carlo simulation
        self.nphot = np.full(self.ncell, nphot)  # Set nphot to initial number.
        self.niter = self.ncell  # Estimated crossing time.
        self.fixseed = self.seed

        t1 = time()
        print("model set-up time = " + str(t1-t0))

    def calc_pops(self):
        if self.mol.part2id:
            ne = self.model.grid['ne']
        else:
            ne = np.zeros(self.model.grid['nh2'].shape)

        vel_grid = np.array([self.model.grid['vr'], self.model.grid['vr'], self.model.grid['vr']]).T # TODO

        _calc_pops(self.fixseed, self.fixset, self.goalsnr, bool(self.mol.part2id), self.model.grid['ra'], self.model.grid['rb'], self.model.grid['nmol'], self.model.grid['nh2'], ne, self.model.grid['doppb'], vel_grid, self.mol.lau, self.mol.lal, self.mol.lcu, self.mol.lcl, self.mol.lcu2, self.mol.lcl2, self.mol.down, self.mol.up, self.mol.down2, self.mol.up2, self.mol.aeinst, self.mol.beinstu, self.mol.beinstl, self.model.tcmb, self.ncell, self.nline, self.nlev, self.dust, self.knu, self.norm, self.cmb, self.nphot, self.minpop, self.outfile)



@jit()
def _calc_pops(fixseed, fixset, goalsnr, part2id, ra, rb, nmol, nh2, ne, doppb, vel_grid, lau, lal, lcu, lcl, lcu2, lcl2, down, up, down2, up2, aeinst, beinstu, beinstl, tcmb, ncell, nline, nlev, dust, knu, norm, cmb, nphot, minpop, outfile):
    print('AMC')
    print('AMC: Starting with FIXSET convergence;'
          'limit=' + str(fixset))
    stage = 1  # 1 = initial phase with fixed photon paths=FIXSET.
    percent = 0
    done = False  # have we finished converging yet?

    pops = np.zeros((nlev, ncell))
    opops = np.zeros((nlev, ncell))
    oopops = np.zeros((nlev, ncell))

    while not done:
        conv = 0
        exceed = 0
        totphot = 0
        totphot2 = 0
        minsnr = 1. / fixset  # smallest number to be counted
        staterr = 0.

        # Loop over all cells.
        for idx in range(ncell):
            phot = np.zeros((nline+2, nphot[idx]))

            # Always do sets of three to build SNR.
            for iternum in range(3):

                # Stage 1=FIXSET > re-initialize random generator each time
                if (stage == 1):
                    np.random.seed(fixseed)

                for ilev in range(nlev):
                    oopops[ilev, idx] = opops[ilev, idx]
                    opops[ilev, idx] = pops[ilev, idx]

                if (nh2[idx] >= eps):
                    #t0 = time()
                    phot = photon(fixseed, stage, ra, rb, nmol, doppb, vel_grid, lau, lal, aeinst, beinstu, beinstl, tcmb, ncell, nline, pops, dust, knu, norm, cmb, nphot[idx], idx)
                    #t1 = time()
                    #print("photon time = " + str(t1-t0))

                    #t0 = time()
                    staterr, pops = stateq(part2id, phot, nmol, nh2, ne, doppb, lau, lal, lcu, lcl, lcu2, lcl2, down, up, down2, up2, aeinst, beinstu, beinstl, nline, nlev, pops, dust, knu, norm, minpop, idx)
                    #t1 = time()
                    #print("stateq time = " + str(t1-t0))

            t0 = time()
            # Determine snr in cell
            snr = fixset
            var = 0.
            totphot += nphot[idx]

            for ilev in range(nlev):
                avepops = pops[ilev, idx] + opops[ilev, idx]
                avepops = (avepops + oopops[ilev, idx]) / 3.
                if avepops >= minpop:
                    var = np.max([np.abs(pops[ilev, idx] - avepops)/avepops, np.abs(opops[ilev, idx] - avepops)/avepops, np.abs(oopops[ilev, idx] - avepops)/avepops])
                    snr = np.max([snr, var])
            snr = 1./snr
            minsnr = np.min([snr, minsnr])

            if (stage == 1):
                if (snr >= 1./fixset):
                    conv += 1    # Stage 1=FIXSET
            else:
                if (snr >= goalsnr):
                    conv += 1
                else:
                    # Double photons if cell not converged.
                    newphot = nphot[idx]*2
                    if (newphot >= max_phot):
                        newphot = max_phot
                        exceed += 1
                        print('AMC: *** Limiting nphot in cell', str(idx))
                    nphot[idx] = newphot

            totphot2 += nphot[idx]
            t1 = time()

        # Report any convergence problems if they occurred
        if (staterr > 0.):
            print('### WARNING: stateq did not converge everywhere'
                  '(err=' + str(staterr) + ')')

        if (stage == 1):
            percent = float(conv)/ncell*100.
            #blowpops(outfile, sim, snr, percent)
            print('AMC: FIXSET fractional error' + " " + str(1./minsnr) + ', ' +
                  str(percent) + '% converged')
            if (conv == ncell):
                stage = 2
                print('AMC')
                print('AMC: FIXSET convergence reached...starting RANDOM')
                print('AMC:')
                print('AMC: minimum S/N  |  converged  |     photons  |  increase to')
                print('AMC: -------------|-------------|--------------|-------------')
            # Next iteration
            continue

        else:
            if (conv == ncell):
                percent = 100.
            else:
                if (exceed < ncell):
                    percent = float(conv)/ncell*100.
                    #blowpops(outfile, sim, snr, percent)
                    print('AMC: ' + str(minsnr) + '  |  ' + str(percent) + '% |  ' + str(totphot) + '  |  ' + str(totphot2))
                    # Next iteration
                    continue
                else:
                    print('### WARNING: Insufficient photons.'
                          'Not converged.')

        done = True

    # Convergence reached (or bailed out)
    print('AMC: ' + str(minsnr) + '  |  ' + str(percent) + '% |  ' + str(totphot) + '  |  converged')
    print('AMC:')
    #blowpops(outfile, sim, snr, percent)
    print('AMC: Written output to ' + str(outfile))
