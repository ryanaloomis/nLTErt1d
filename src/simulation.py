import numpy as np
from model import model
from molecule import molecule
from stateq import stateq
from blowpops import blowpops
from time import time
import scipy.constants as sc
from numba import jit

# Define some global variables.

EPS = 1.e-30        # number of photons.
MAX_PHOT = 100000   # max number of photons.
NEGTAULIM = -30.    # negative optical depth limit.
DELTA = 1.0e-10     # delta?
CALC_POPS_ITER = 3  # Number of iterations for calc_pops.
hpip = sc.h * sc.c / 4. / np.pi / np.sqrt(np.pi)


class simulation:
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
        blending (optional [bool]): Whether to include line blending.
        nchan (optional [int]): Number of channels per trans for the ray
            tracing.
        rt_lines (optional [int list]): List of transitions to raytrace.
        velres (optional [float]): Channel res for raytracing (km/s).
    """

    def __init__(self, source, outfile, molfile, goalsnr, nphot, kappa=None,
                 tnorm=2.735, velocity_function=None, seed=1971, minpop=1e-4,
                 fixset=1e-6, blending=False, verbose=True, blend_limit=1e3):
        """Instantiate the class."""

        # Setting simulation properties.
        self.source = source
        self.outfile = outfile
        self.molfile = molfile
        self.goalsnr = goalsnr
        self.kappa_params = kappa
        self.tnorm = tnorm
        self.seed = seed
        self.minpop = minpop
        self.fixset = fixset
        self.blending = blending
        # self.nchan = nchan
        # self.rt_lines = rt_lines
        # self.velres = velres * 1000.  # convert to m/s
        self.verbose = verbose
        # self.velax = self.make_velax()

        t0 = time()
        # Read in the source model (default is RATRAN).
        self.model = model(self.source, 'ratran')
        self.ncell = self.model.ncell

        # Have user input velocity function.
        if velocity_function is not None:
            self.model.velo = simulation.import_velocity(velocity_function)

        # Read in the LAMDA molecular data file
        try:
            self.mol = molecule(self, self.molfile)
        except Exception:
            raise Exception("Couldn't parse molecular data.")
        self.blends = self.get_blends(blend_limit)

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

        # Get the collisional rates at each cell.
        # Should have shape (nlines x ncells).
        rates = self.mol.get_rates(self.model.tkin, fill_value='extrapolate')
        self.up1, self.down1, self.up2, self.down2 = rates

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
                self.knu[l, i] *= self.model.nh2[i] / self.model.g2d
                self.dust[l, i] = simulation.planck(self.mol.freq[l],
                                                    self.model.tdust[i])

        # Set up the Monte Carlo simulation
        self.nphot = np.full(self.ncell, nphot)  # Set nphot to initial number.
        self.niter = self.ncell  # Estimated crossing time.
        self.fixseed = self.seed

        # Setup time.
        if self.verbose:
            print("Set up took {:.1f} ms.".format((time() - t0) * 1e3))

    def get_blends(self, blend_limit=1.0e3):
        """
        Return all the possible line blends. If there are no blends found but
        `blending` has been set to `True`, turn it off to speed up the run.

        Args:
            dv_limit (optional[float]): Difference between line centres in
                [m/s] which are considered to be blended.

        Returns:
            An array of the blended components with a shape of `(nblends, 3)`,
            with each blend having the two line indices and the velocity
            separation between them in [m/s].
        """
        blends = self.mol.get_blends(blend_limit=blend_limit)
        if blends.shape[0] == 0 and self.blending is False:
            if self.verbose:
                print("WARNING: Blends have been found." +
                      "You may want to turn it on with `blending=True`.")
        if blends.shape[0] == 0 and self.blending is True:
            if self.verbose:
                print("No blends have been found. Turning off blending.")
            self.blending = False
        return blends if blends.shape[0] != 0 else np.zeros((1, 3))

    def calc_pops(self):
        """Wrapper for the Numba calculation of populations function."""
        output = _calc_pops(fixseed=self.fixseed,
                            fixset=self.fixset,
                            goalsnr=self.goalsnr,
                            part2id=bool(self.mol.part2id),
                            ra=self.model.ra,
                            rb=self.model.rb,
                            nmol=self.model.nmol,
                            nh2=self.model.nh2,
                            ne=self.model.ne,
                            doppb=self.model.doppb,
                            vel_grid=self.model.velocities,
                            lau=self.mol.lau,
                            lal=self.mol.lal,
                            lcu=self.mol.lcu,
                            lcl=self.mol.lcl,
                            lcu2=self.mol.lcu2,
                            lcl2=self.mol.lcl2,
                            down=self.down1,
                            up=self.up1,
                            down2=self.down2,
                            up2=self.up2,
                            aeinst=self.mol.aeinst,
                            beinstu=self.mol.beinstu,
                            beinstl=self.mol.beinstl,
                            blending=self.blending,
                            blends=self.blends,
                            tcmb=self.model.tcmb,
                            ncell=self.ncell,
                            nline=self.nline,
                            nlev=self.nlev,
                            ntrans=self.ntrans,
                            ntrans2=self.ntrans2,
                            dust=self.dust,
                            knu=self.knu,
                            norm=self.norm,
                            cmb=self.cmb,
                            nphot=self.nphot,
                            minpop=self.minpop,
                            outfile=self.outfile,
                            eps=EPS,
                            max_phot=MAX_PHOT,
                            verbose=self.verbose)
        self.pops, self.snr, self.percent = output
        blowpops(self.outfile, self, self.snr, self.percent)
        if self.verbose:
            print('AMC: Written output to ' + self.outfile)

    def raytrace(self, v_min, v_max, nv=None, dv=None, rt_lines=[1, 2, 3]):
        """
        Generate a spectrum given the specified parameters.

        Args:
            v_min (float):
            v_max (float):
            nv (optional[int]):
            dv (optional[float]):
            rt_lines (optional[iterable]):

        Returns:
            TBD
        """

        # Make the velocity axis.
        velax = simulation.make_velax(v_min, v_max, nv, dv)
        nchan, velres = len(velax), np.diff(velax)[0]

        # Perform the ray tracing.
        intens, tau = _raytrace(self.model.rmax, self.model.ra,
                                self.model.rb, self.model.nmol,
                                self.model.nh2, self.model.doppb,
                                self.model.velocities, self.mol.lau,
                                self.mol.lal, self.mol.aeinst,
                                self.mol.beinstu, self.mol.beinstl,
                                self.blending, self.blends,
                                self.model.tcmb, self.ncell, self.pops,
                                self.dust, self.knu, self.norm, self.cmb,
                                nchan, int(nchan/2.), velres, rt_lines)

        # Convert to [K] using X law.
        ucon = sc.c**2 / self.mol.freq[rt_lines]**2 / 2. / sc.k
        intens = intens * ucon[:, None] * self.norm[rt_lines][:, None]
        return velax, intens, tau

    @staticmethod
    def make_velax(v_min, v_max, nv=None, dv=None):
        """Generate the velocity axis."""
        if v_max < v_min:
            v_min, v_max = v_max, v_min
        if nv is not None and dv is not None:
            raise ValueError("Must specify either `nx` or `dx`, not both.")
        if nv is not None:
            v = np.linspace(v_min, v_max, nv)
        elif dv is not None:
            v = np.arange(v_min, v_max+dv, dv)
            v = v if v[-1] == v_max else v[:-1]
            v += 0.5 * (v_max - v[-1])
        else:
            raise ValueError("Must specify either `nx` or `dx`, not both.")
        return v

    def velax_to_freqax(self, velax, nu0=None, line_idx=None, vlsr=0.0):
        """
        Convert a velocity axis to a frequency axis.

        Args:
            velax (array): Velocity axis in [m/s].
            nu0 (optional[float]): Rest frequency in [Hz].
            line_idx (optional[int]): Index of the line to observe.
            vlsr (optional[float]): Systemic velocity in [m/s].

        Returns:
            The frequency axis in [Hz].
        """
        if nu0 is not None and line_idx is not None:
            raise ValueError("Must specify `nu0` or `line_idx`, not both.")
        if line_idx is not None:
            nu0 = self.mol.freq[line_idx]
        if nu0 is None:
            raise ValueError("Must specify `nu0` or `line_idx`, not both.")
        return nu0 * (1. - (velax - vlsr) / sc.c)

    @staticmethod
    def planck(freq, temp, eps=EPS):
        """
        Planck function for a given frequency [Hz] and temperature [K].

        Args:
            freq (float): Frequency in [Hz].
            temp (float): Temperature in [K].

        Returns:
            Bnu (float): Spectral radiance in [W/m^2/sr^1/Hz]. If the
                temperature is too low (T < EPS), returns zero.
        """
        Bnu = 2. * sc.h * freq**3 * sc.c**-2
        Bnu /= np.exp(sc.h * freq / (sc.k * temp)) - 1.0
        return np.where(temp >= eps, Bnu, 0.0)

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
            filename = "kappa/jena_{}_{}.tab".format(params[1], params[2])
            filename = "../src/" + filename
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


# -- Functions for Numba -- #

# @jit()
def _calc_pops(fixseed, fixset, goalsnr, part2id, ra, rb, nmol, nh2, ne, doppb,
               vel_grid, lau, lal, lcu, lcl, lcu2, lcl2, down, up, down2, up2,
               aeinst, beinstu, beinstl, blending, blends, tcmb, ncell, nline,
               nlev, ntrans, ntrans2, dust, knu, norm, cmb, nphot, minpop,
               outfile, eps, max_phot, verbose=True):
    """
    Docstring coming.
    """

    if verbose:
        print('AMC')
        print('AMC: Starting with FIXSET convergence;')
        # print('limit = {}'.format(fixset))

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
            for iternum in range(CALC_POPS_ITER):

                # Stage 1=FIXSET > re-initialize random generator each time
                if (stage == 1):
                    np.random.seed(fixseed)

                for ilev in range(nlev):
                    oopops[ilev, idx] = opops[ilev, idx]
                    opops[ilev, idx] = pops[ilev, idx]

                if (nh2[idx] >= eps):
                    phot = photon(fixseed, stage, ra, rb, nmol, doppb,
                                  vel_grid, lau, lal, aeinst, beinstu, beinstl,
                                  blending, blends, tcmb, ncell, nline, pops,
                                  dust, knu, norm, cmb, nphot[idx], idx)

                    staterr, pops = stateq(part2id, phot, nmol, nh2, ne, doppb,
                                           lau, lal, lcu, lcl, lcu2, lcl2,
                                           down, up, down2, up2, aeinst,
                                           beinstu, beinstl, blending, blends,
                                           nline, nlev, ntrans, ntrans2, pops,
                                           dust, knu, norm, minpop, idx)

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
                if (snr >= 1. / fixset):
                    conv += 1    # Stage 1=FIXSET
            else:
                if (snr >= goalsnr):
                    conv += 1
                else:
                    # Double photons if cell not converged.
                    newphot = nphot[idx] * 2
                    if (newphot >= max_phot):
                        newphot = max_phot
                        exceed += 1
                        print('AMC: *** Limiting nphot in cell', str(idx))
                    nphot[idx] = newphot

            totphot2 += nphot[idx]

        # Report any convergence problems if they occurred
        if (staterr > 0.):
            print('### WARNING: stateq did not converge everywhere'
                  '(err = {})'.format(staterr))

        if (stage == 1):
            percent = 100. * float(conv) / ncell
            print('AMC: FIXSET fractional error {:.3e}, '.format(1. / minsnr)
                  + '{:.2f}% converged.'.format(percent))
            if (conv == ncell):
                stage = 2
                print('AMC')
                print('AMC: FIXSET convergence reached...starting RANDOM')
                print('AMC:')
                print('AMC:  min S/N |  conv  | photons | increase to ')
                print('AMC: ---------|--------|---------|-------------')
            # Next iteration
            continue

        else:
            if (conv == ncell):
                percent = 100.
            else:
                if (exceed < ncell):
                    percent = float(conv) / ncell * 100.
                    print('AMC: ' +
                          ' {:6.3f}  |'.format(minsnr) +
                          ' {:4.2f}% |'.format(percent) +
                          ' {:6d}  |'.format(totphot) +
                          ' {:6d}'.format(totphot2))
                    # Next iteration
                    continue
                else:
                    print('### WARNING: Insufficient photons.\nNot converged.')

        done = True

    # Convergence reached (or bailed out)
    print('AMC:  {:6.3f}  |'.format(minsnr) +
          ' {:4.1f}% |'.format(percent) +
          ' {:6d}  |'.format(totphot) +
          ' converged')
    print('AMC:')

    return pops, snr, percent


# @jit()
def _raytrace(rmax, ra, rb, nmol, nh2, doppb, vel_grid, lau, lal, aeinst,
              beinstu, beinstl, blending, blends, tcmb, ncell, pops, dust,
              knu, norm, cmb, nchan, vcen, velres, rt_lines):
    """
    Line of sight integration.

    Args:
        TBD

    Returns:
        TBD
    """

    # Set global parameters.

    reflect = True
    nlines = len(rt_lines)

    # Empty arrays for Numba.

    tau = np.zeros((nlines, nchan))
    intens = np.zeros((nlines, nchan))

    # Raytrace photons along cells, with photon velocity determined by
    # channel step size. Start at front of source

    rpos = rmax * (1.0 - DELTA)
    posn = ncell - 1

    phi = -np.pi
    cosphi = np.cos(phi)

    # Propagate to edge of cloud by moving from cell edge to cell edge.
    # After leaving the source, add CMB and store intensities

    in_cloud = True
    while in_cloud:

        # Find distance to nearest cell edge
        rnext = ra[posn] * (1.0 - DELTA)

        bac = 4. * ((rpos * cosphi)**2 - rpos**2 + rnext**2)
        dsplus = -0.5 * (2. * rpos * cosphi + np.sqrt(bac))
        dsminn = -0.5 * (2. * rpos * cosphi - np.sqrt(bac))

        if (dsminn * dsplus == 0.):
            ds = 0.
        else:
            if (dsplus < 0.):
                ds = dsminn
            if (dsminn < 0.):
                ds = dsplus
            if (dsminn*dsplus > 0.):
                ds = np.min([dsplus, dsminn])

        # Find "vfac", the velocity line profile factor
        # Number of splines nspline=los_delta_v/local_line_width
        # Number of averaging steps naver=local_delta_v/local_line_width

        if (nmol[posn] > EPS):
            for ichan in range(nchan):
                deltav = (ichan - vcen) * velres
                b = doppb[posn]
                vfac = _calcLineAmp(b, vel_grid, rpos, ds, phi, deltav, posn)

                for idx, iline in enumerate(rt_lines):
                    # backwards integrate dI/ds

                    jnu = vfac * hpip * nmol[posn] * aeinst[iline] / b
                    jnu *= pops[lau, posn][iline]
                    jnu += dust[iline, posn] * knu[iline, posn]

                    alpha = pops[lal, posn][iline] * beinstl[iline]
                    alpha -= pops[lau, posn][iline] * beinstu[iline]
                    alpha *= vfac * hpip / b * nmol[posn]
                    alpha += knu[iline, posn]
                    dtau = max(NEGTAULIM, alpha * ds)

                    if abs(alpha) < EPS:
                        snu = 0.0
                    else:
                        snu = jnu / alpha / norm[iline]
                    intens_tmp = snu * (1. - np.exp(-dtau))
                    intens_tmp *= np.exp(-tau[idx, ichan])

                    intens[idx, ichan] += intens_tmp
                    tau[idx, ichan] += dtau

                    # Line blending step

                    if blending:
                        for iblend in range(blends.shape[0]):
                            if iline == int(blends[iblend, 0]):

                                jl = int(blends[iblend, 1])
                                bdeltav = blends[iblend, 2]

                                velproj = deltav - bdeltav
                                bvfac = _calcLineAmp(b, vel_grid, rpos, ds,
                                                     phi, velproj, posn)

                                bjnu = bvfac * hpip * nmol[posn] / b
                                bjnu *= aeinst[jl] * pops[lau, posn][jl]

                                balpha = pops[lal, posn][jl] * beinstl[jl]
                                balpha -= pops[lau, posn][jl] * beinstu[jl]
                                balpha *= bvfac * hpip * nmol[posn] / b

                                if abs(balpha) < EPS:
                                    bsnu = 0.
                                else:
                                    bsnu = bjnu / balpha / norm[jl]
                                bdtau = max(NEGTAULIM, balpha * ds)

                                intens_tmp = np.exp(-tau[idx, ichan])
                                intens_tmp *= bsnu * (1. - np.exp(-bdtau))
                                intens[idx, ichan] += intens_tmp
                                tau[idx, ichan] += bdtau

        # Update photon position, direction; check if escaped
        posn -= 1
        if (posn < 0 or posn >= ncell):
            break        # reached edge of cloud, break

        rpos = rnext

    # if reflecting, propogate through the whole cloud again in the other
    # direction (but now velocities are reversed)
    if reflect is True:
        rpos = DELTA
        phi = np.pi
        posn = 0
        cosphi = np.cos(phi)

        while in_cloud:
            # Find distance to nearest cell edge
            # print(posn)
            rnext = rb[posn]*(1 + DELTA)
            bac = 4.*((rpos*cosphi)**2. - rpos**2. + rnext**2.)

            dsplus = 0.5*(2.*rpos*cosphi + np.sqrt(bac))
            dsminn = 0.5*(2.*rpos*cosphi - np.sqrt(bac))

            if (dsminn*dsplus == 0.):
                ds = 0.
            else:
                if (dsplus < 0.):
                    ds = dsminn
                if (dsminn < 0.):
                    ds = dsplus
                if (dsminn*dsplus > 0.):
                    ds = np.min(np.array([dsplus, dsminn]))

            # Find "vfac", the velocity line profile factor
            # Number of splines nspline=los_delta_v/local_line_width
            # Number of averaging steps naver=local_delta_v/local_line_width

            if (nmol[posn] > EPS):
                for ichan in range(nchan):
                    deltav = (ichan - vcen) * velres
                    b = doppb[posn]
                    vfac = _calcLineAmp(b, -vel_grid, rpos, ds, phi, deltav,
                                        posn)

                    for idx, iline in enumerate(rt_lines):
                        # backwards integrate dI/ds

                        jnu = vfac * hpip * nmol[posn] / b
                        jnu *= pops[lau, posn][iline] * aeinst[iline]
                        jnu += dust[iline, posn] * knu[iline, posn]

                        alpha = pops[lal, posn][iline] * beinstl[iline]
                        alpha -= pops[lau, posn][iline] * beinstu[iline]
                        alpha *= vfac * hpip * nmol[posn] / b
                        alpha += knu[iline, posn]

                        snu = jnu / alpha / norm[iline]
                        if np.abs(alpha) < 0:
                            snu = 0.
                        dtau = max(NEGTAULIM, alpha * ds)

                        intens_tmp = np.exp(-tau[idx, ichan])
                        intens_tmp *= (1. - np.exp(-dtau)) * snu
                        intens[idx, ichan] += intens_tmp
                        tau[idx, ichan] += dtau

                        # Line blending step

                        if blending:
                            for iblend in range(blends.shape[0]):
                                if iline == int(blends[iblend, 0]):

                                    jl = int(blends[iblend, 1])
                                    bdeltav = blends[iblend, 2]

                                    velproj = deltav - bdeltav
                                    bvfac = _calcLineAmp(b, vel_grid, rpos, ds,
                                                         phi, velproj, posn)

                                    bjnu = bvfac * hpip * nmol[posn] / b
                                    bjnu *= pops[lau, posn][jl] * aeinst[jl]
                                    balpha = pops[lal, posn][jl] * beinstl[jl]
                                    balpha -= pops[lau, posn][jl] * beinstu[jl]
                                    balpha *= bvfac * hpip / b * nmol[posn]

                                    if abs(balpha) < EPS:
                                        bsnu = 0.
                                    else:
                                        bsnu = bjnu/balpha/norm[jl]

                                    bdtau = balpha*ds
                                    if bdtau < NEGTAULIM:
                                        bdtau = NEGTAULIM

                                    intens_tmp = np.exp(-tau[idx, ichan])
                                    intens_tmp *= bsnu * (1. - np.exp(-bdtau))
                                    intens[idx, ichan] += intens_tmp
                                    tau[idx, ichan] += bdtau

            # Update photon position, direction; check if escaped
            posn += 1
            if (posn < 0 or posn >= ncell):
                break
            rpos = rnext

    # Finally, add cmb to memorized i0 incident on cell id
    if (tcmb > 0.):
        for idx, iline in enumerate(rt_lines):
            intens[idx, :] += np.exp(-tau[idx, :]) * cmb[iline]
    return intens, tau


@jit(nopython=True)
def photon(fixseed, stage, ra, rb, nmol, doppb, vel_grid, lau, lal, aeinst,
           beinstu, beinstl, blending, blends, tcmb, ncell, nline, pops, dust,
           knu, norm, cmb, nphot, idx):
    """What?"""

    phot = np.zeros((nline+2, nphot))
    if stage == 1:
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

        vel = vel_grid[:, idx]
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
                rnext = rb[posn]*(1 + DELTA)
                bac = 4.*((rpos*cosphi)**2. - rpos**2. + rnext**2.)
            else:
                dpos = -1
                rnext = ra[posn]*(1 - DELTA)
                bac = 4.*((rpos*cosphi)**2. - rpos**2. + rnext**2.)
                if (bac < 0.):
                    dpos = 1
                    rnext = rb[posn]*(1 + DELTA)
                    bac = 4.*((rpos*cosphi)**2. - rpos**2. + rnext**2.)

            dsplus = -0.5*(2.*rpos*cosphi + np.sqrt(bac))
            dsminn = -0.5*(2.*rpos*cosphi - np.sqrt(bac))

            if (dsminn*dsplus == 0.):
                ds = 0.
            else:
                if (dsplus < 0.):
                    ds = dsminn
                if (dsminn < 0.):
                    ds = dsplus
                if (dsminn*dsplus > 0.):
                    ds = np.min(np.array([dsplus, dsminn]))

            # Find "vfac", the velocity line profile factor
            # Number of splines nspline=los_delta_v/local_line_width
            # Number of averaging steps naver=local_delta_v/local_line_width

            if (nmol[posn] > EPS):
                b = doppb[posn]
                vfac = _calcLineAmp(b, vel_grid, rpos, ds, phi, deltav, idx)

                # backwards integrate dI/ds
                jnu = dust[:,posn]*knu[:,posn] + vfac*hpip/b*nmol[posn]*pops[lau,posn]*aeinst
                alpha = knu[:,posn] + vfac*hpip/b*nmol[posn]*(pops[lal,posn]*beinstl - pops[lau,posn]*beinstu)

                snu = jnu/alpha/norm
                snu[np.abs(alpha) < EPS] = 0.

                dtau = alpha*ds
                dtau[dtau < NEGTAULIM] = NEGTAULIM

                if not firststep:
                    phot[2:, iphot] += np.exp(-tau)*(1. - np.exp(-dtau))*snu
                    tau += dtau
                    tau[tau < NEGTAULIM] = NEGTAULIM


                # Line blending step - note that this is not vectorized as the previous calculations have been
                if blending:
                    for iblend in range(blends.shape[0]):
                        bjnu = 0.
                        balpha = 0.
                        iline = int(blends[iblend,0])
                        jline = int(blends[iblend,1])
                        bdeltav = blends[iblend,2]

                        velproj = deltav - bdeltav
                        bvfac = _calcLineAmp(b, vel_grid, rpos, ds, phi, velproj, idx)

                        bjnu = bvfac*hpip/b*nmol[posn]*pops[lau,posn][jline]*aeinst[jline]
                        balpha = bvfac*hpip/b*nmol[posn]*(pops[lal,posn][jline]*beinstl[jline] - pops[lau,posn][jline]*beinstu[jline])

                        if np.abs(balpha) < EPS:
                            bsnu = 0.
                        else:
                            bsnu = bjnu/balpha/norm[jline]

                        bdtau = balpha*ds
                        if bdtau < NEGTAULIM:
                            bdtau = NEGTAULIM

                        if not firststep:
                            dphot = np.exp(-tau[iline]) * (1. - np.exp(-bdtau))
                            phot[2 + iline, iphot] += dphot * bsnu
                            tau[iline] += bdtau
                            tau[tau < NEGTAULIM] = NEGTAULIM

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


@jit(nopython=True)
def _vfunc(v, s, rpos, phi, vphot):
    """Projected velocity difference between photon and gas."""
    psis = np.arctan2(s * np.sin(phi), rpos + s * np.cos(phi))
    return vphot - np.cos(phi - psis) * v[0]


@jit(nopython=True)
def _calcLineAmp(b, vel_grid, rpos, ds, phi, deltav, idx):
    """Calculate the line amplitude."""
    v1 = _vfunc(vel_grid[:, idx], 0., rpos, phi, deltav)
    v2 = _vfunc(vel_grid[:, idx], ds, rpos, phi, deltav)
    nspline = np.maximum(1, int(np.abs(v1 - v2)/b))
    vfac = 0.
    for ispline in range(nspline):
        s1 = ds * ispline / nspline
        s2 = ds * (ispline+1.) / nspline
        v1 = _vfunc(vel_grid[:, idx], s1, rpos, phi, deltav)
        v2 = _vfunc(vel_grid[:, idx], s2, rpos, phi, deltav)
        naver = np.maximum(1, int(np.abs(v1 - v2)/b))
        for iaver in range(naver):
            s = s1 + (s2 - s1) * (iaver + 0.5) / naver
            v = _vfunc(vel_grid[:, idx], s, rpos, phi, deltav)
            vfacsub = np.exp(-(v / b)**2)
            vfac += vfacsub / naver
    vfac /= nspline
    return vfac
