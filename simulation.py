import numpy as np
from model import model
from molecule import molecule
from photon_1d import photon
from stateq import stateq
# from blowpops import blowpops
from time import time
import scipy.constants as sc
from numba import jit


class simulation:
    """The main simulation class."""

    eps = 1.e-30        # number of photons.
    max_phot = 100000   # max number of photons.
    negtaulim = -30.    # negative optical depth limit.
    delta = 1.0e-10     # delta?

    def __init__(self, source, outfile, molfile, goalsnr, nphot, kappa=None,
                 tnorm=2.735, velocity_function=None, seed=1971, minpop=1e-4,
                 fixset=1.e-6, blending=False, debug=True):
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
        # self.velocity = velocity
        self.seed = seed
        self.minpop = minpop
        self.fixset = fixset
        self.blending = blending
        self.debug = debug

        t0 = time()
        # Read in the source model (default is RATRAN).
        if self.debug:
            print('[debug] reading in source model')
        self.model = model(self.source, 'ratran', self.debug)
        self.ncell = self.model.ncell

        # Have user input velocity function.
        print(velocity_function)
        if velocity_function is not None:
            if self.debug:
                print("[debug] Importing user velocity function.")
            self.model.velo = simulation.import_velocity(velocity_function)

        # Read in the molfile
        if self.debug:
            print('[debug] reading molecular data file')

        try:
            self.mol = molecule(self, self.molfile)
        except:
            raise Exception("Couldn't parse molecular data.")

        self.nlev = self.mol.nlev
        self.nline = self.mol.nline
        self.ntrans = self.mol.ntrans
        self.ntrans2 = self.mol.ntrans2
        self.ntemp = self.mol.ntemp
        self.ntemp2 = self.mol.ntemp2

        # Include thermal broadening the to widths.
        v_turb2 = self.model.doppb**2.0
        v_therm2 = 2 * sc.k * self.model.tkin / sc.m_p / self.mol.molweight
        self.model.doppb = np.sqrt(v_turb2 + v_therm2)

        # Calculate the collisional rates.
        self.mol.set_rates(self.model.tkin)
        if self.mol.up is None or self.mol.down is None:
            raise ValueError("Need to calculate rates.")

        # Initialize dust emissivity, convertiong from [m^2/kg] to [m^-1/n(H2)]
        # such that tau_dust = knu.
        self.norm = simulation.planck(self.mol.freq[0], self.tnorm)
        self.norm = self.norm * np.ones(self.nline)
        self.cmb = simulation.planck(self.mol.freq, self.model.tcmb)
        self.cmb /= self.norm

        # Parse kappa parameters and generate the kappa function
        self.kappa = simulation.generate_kappa(self.kappa_params)

        # Do not normalize dust; will be done in photon.
        # Funky looping as functions fail to broadcast.
        self.knu = np.zeros((self.nline, self.ncell))
        self.dust = np.zeros((self.nline, self.ncell))
        for l in range(self.nline):
            for i in range(self.ncell):
                self.knu[l, i] = self.kappa(i, self.mol.freq[l]) * 2.4 * sc.m_p
                self.knu[l, i] /= self.model.g2d * self.model.nh2[i]
                self.dust[l, i] = simulation.planck(self.mol.freq[l],
                                                    self.model.tdust[i])

        # Set up the Monte Carlo simulation
        self.nphot = np.full(self.ncell, nphot)  # Set nphot to initial number.
        self.niter = self.ncell  # Estimated crossing time.
        self.fixseed = self.seed

        t1 = time()
        print("Set up took %.1f ms." % ((t1 - t0) * 1e3))

    def calc_pops(self):
        """Wrapper for the calculation of populations function."""
        _calc_pops(self.fixseed, self.fixset, self.goalsnr,
                   bool(self.mol.part2id), self.model.ra,
                   self.model.rb, self.model.nmol,
                   self.model.nh2, self.model.ne, self.model.doppb,
                   self.model.velocities, self.mol.lau, self.mol.lal,
                   self.mol.lcu, self.mol.lcl, self.mol.lcu2, self.mol.lcl2,
                   self.mol.down, self.mol.up, self.mol.down2, self.mol.up2,
                   self.mol.aeinst, self.mol.beinstu, self.mol.beinstl,
                   self.blending, self.mol.blends,
                   self.model.tcmb, self.ncell, self.nline, self.nlev,
                   self.dust, self.knu, self.norm, self.cmb, self.nphot,
                   self.minpop, self.outfile, simulation.eps,
                   simulation.max_phot)

    @staticmethod
    def planck(freq, temp):
        """
        Planck function for a given frequency [Hz] and temperature [K].

        Args:
            freq (float): Frequency in [Hz].
            temp (float): Temperature in [K].

        Returns:
            Bnu (float): Spectral radiance in [W/m^2/sr^1/Hz]. If the
                temperature is too low (T < eps), returns zero.
        """
        Bnu = 2. * sc.h * freq**3 * sc.c**-2
        Bnu /= np.exp(sc.h * freq / (sc.k * temp)) - 1.0
        return np.where(temp >= simulation.eps, Bnu, 0.0)

    @staticmethod
    def generate_kappa(kappa_params=None):
        """
        Returns the dust emissivity in [m^2/kg] at a frequency and in cell ID.
        TODO: Make the input directly file paths or floats?

        Args:
            kappa_params (str): A string decribing the dust parameters. For the
                use of Ossenkopf & Henning (1994) opacities it must take the
                form of:

                    kappa_params = 'jena, TYPE, COAG'

                where TYPE must be 'bare', 'thin' or 'thick' and COAG must be
                'no', 'e5', 'e6', 'e7' or 'e8'. Otherwise a power law profile
                can be included. Alternatively, a simple power law can be used
                where the parameters are given by:

                    kappa_params = 'powerlaw, freq0, kappa0, beta'

                where freq0 is in [Hz], kappa0 in [cm^2/g], and beta is the
                frequency index. If nothing is given, we assume no opacity.

        Returns:
            kappa: A callable function returning the dust emissivity in
                [m^2/kg].

        Raises:
            ValueError: If the parameters cannot be parsed.
        """

        # If no parameters are provided, assume no dust opacity.
        if kappa_params is None:
            def kappa(idx, freq):
                return 0.0
            return kappa

        # Parse the input parameters.
        params = kappa_params.replace(' ', '').lower().split(',')

        # Ossenkopf & Henning (199X) dust opacities.
        if params[0] == 'jena':
            from scipy.interpolate import CubicSpline
            filename = "kappa/jena_" + params[1] + "_" + params[2] + ".tab"
            table = np.loadtxt(filename)
            lamtab = np.log10(table[:, 0]) - 6.0
            kaptab = np.log10(table[:, 1])
            interp_func = CubicSpline(lamtab, kaptab, extrapolate=True)

            # The 0.1 converts [cm^2/g] to [m^2/kg].
            def kappa(idx, freq):
                lam_lookup = np.log10(sc.c / freq)
                return 0.1 * 10**interp_func(lam_lookup)
            return kappa

        # Simple power law opacity.
        elif params[0] == 'powerlaw':
            freq0 = float(params[1])
            kappa0 = float(params[2])
            beta = float(params[3])

            # The 0.1 converts [cm^2/g] to [m^2/kg].
            def kappa(idx, freq):
                return 0.1 * kappa0 * (freq / freq0)**beta
            return kappa
        else:
            raise ValueError("Invalid kappa_params.")

    @staticmethod
    def import_velocity(velocity_function):
        """
        Import a velocity function in the provided file.
        Args:
            velocity_function (tuple): Relative path to the file containing the
                function and the function name, for example:
                    '~/folders/functions.py, velocity_function'
        Returns:
            velo (function): Callable function.
        """
        import os
        import sys
        velo = None
        path, function_name = velocity_function.replace(' ', '').split(',')
        file_directory = '/'.join(os.path.expanduser(path).split('/')[:-1])
        file_name = path.split('/')[-1].replace('.py', '')
        sys.path.append(file_directory)
        exec('from %s import %s as velo' % (file_name, function_name))
        return velo


@jit()
def _calc_pops(fixseed, fixset, goalsnr, part2id, ra, rb, nmol, nh2, ne, doppb,
               vel_grid, lau, lal, lcu, lcl, lcu2, lcl2, down, up, down2, up2,
               aeinst, beinstu, beinstl, blending, blends, tcmb, ncell, nline, 
               nlev, dust, knu, norm, cmb, nphot, minpop, outfile, eps, 
               max_phot):
    """
    Docstring coming.
    """
    print('AMC')
    print('AMC: Starting with FIXSET convergence;\nlimit=' + str(fixset))
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
                    phot = photon(fixseed, stage, ra, rb, nmol, doppb, vel_grid, lau, lal, aeinst, beinstu, beinstl, blending, blends, tcmb, ncell, nline, pops, dust, knu, norm, cmb, nphot[idx], idx)
                    #t1 = time()
                    #print("photon time = " + str(t1-t0))

                    #t0 = time()
                    staterr, pops = stateq(part2id, phot, nmol, nh2, ne, doppb, lau, lal, lcu, lcl, lcu2, lcl2, down, up, down2, up2, aeinst, beinstu, beinstl, blending, blends, nline, nlev, pops, dust, knu, norm, minpop, idx)
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
                    var = np.max([np.abs(pops[ilev, idx] - avepops)/avepops,
                                  np.abs(opops[ilev, idx] - avepops)/avepops,
                                  np.abs(oopops[ilev, idx] - avepops)/avepops])
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
            # blowpops(outfile, sim, snr, percent)
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
