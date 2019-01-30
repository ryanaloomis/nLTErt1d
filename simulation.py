import numpy as np
from model import model
from molecule import molecule
from common import *
from numerical import *
from kappa import generate_kappa
from photon_1d import photon
from stateq import *

class simulation:
    def __init__(self, source, outfile, molfile, goalsnr, nphot, kappa=None, tnorm=100, velocity='grid', seed=1971, minpop=1e-4, fixset=1.e-6, debug=False):
        # Init user specified parameters
        self.source = source
        self.outfile = outfile
        self.molfile = molfile
        self.goalsnr = goalsnr
        # not setting nphot yet, setting later as array w/ size ncell
        self.kappa_params = kappa
        self.tnorm = tnorm
        self.seed = seed
        self.minpop = minpop
        self.fixset = fixset
        self.debug = debug

        # Read in the source model
        if self.debug: print('[debug] reading in source model')
        self.model = model(self.source, self.debug)
        self.ncell = self.model.ncell

        # Check to make sure that there's a valid velocity field
        if velocity == 'grid':
            if 'vr' not in self.model.grid:
                raise Exception('ERROR: Velicity mode specified as grid, but no valid velocity field included in model.')
        else:
            try:
                # Read in user velocity function and override the grid lookup function
                velo_file = velocity
                from velo_file import velo as user_velo
                self.model.velo = user_velo
            except:
                raise Exception('ERROR: Could not read velocity function from file ' + velo_file)

        # Read in the molfile
        if self.debug: print('[debug] reading molecular data file')
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
                self.cmb[iline] = planck(self.mol.freq[iline], self.model.tcmb)/self.norm[iline]

        # Parse kappa parameters and generate the kappa function
        self.kappa = generate_kappa(self.kappa_params)

        # Do not normalize dust; will be done in photon
        self.knu = np.zeros((self.nline, self.ncell))
        self.dust = np.zeros((self.nline, self.ncell))
        for iline in range(self.nline):
            for idx in range(self.ncell):
                self.knu[iline, idx] = self.kappa(idx, self.mol.freq[iline])*2.4*amu/self.model.gas2dust*self.model.grid['nh2'][idx]
                self.dust[iline,idx] = planck(self.mol.freq[iline],self.model.grid['tdust'][idx])

        # Set up the Monte Carlo simulation
        self.pops = np.zeros((self.nlev, self.ncell))
        self.opops = np.zeros((self.nlev, self.ncell))
        self.oopops = np.zeros((self.nlev, self.ncell))
        self.nphot = np.full(self.ncell, nphot)         # Set nphot to initial number
        self.niter = self.ncell                           # Estimated crossing time
        self.fixseed = self.seed
        self.phot = np.zeros((self.nline+2, nphot))
        self.tau = np.zeros(self.nline)



    def calc_pops(self):
        print('AMC')
        print('AMC: Starting with FIXSET convergence; limit=' + str(self.fixset))

        stage = 1                           # 1=initial phase with fixed photon paths=FIXSET
        percent = 0
        done = False                        # have we finished converging yet?

        while done==False:
            conv = 0
            exceed = 0
            totphot = 0
            totphot2 = 0
            minsnr=1./self.fixset                 # fixset is smallest number to be counted
            staterr = 0.

            for idx in range(self.ncell):        # Loop over all cells
                for iternum in range(3):    # always do sets of 3 iterations to build snr

                    if (stage == 1):        # Stage 1=FIXSET -> re-initialize ran1 each time
                        np.random.seed(self.fixseed)
                        dummy = np.random.random()
                    
                    for ilev in range(self.nlev):
                        self.oopops[ilev, idx] = self.opops[ilev, idx]
                        self.opops[ilev, idx] = self.pops[ilev, idx]

                    if (self.model.grid['nh2'][idx] >= eps):
                        if self.debug: print('[debug] calling photon for cell ' + str(idx))
                        photon(self, idx, self.debug)

                        if self.debug: print('[debug] calling stateq for cell ' + str(idx))
                        staterr = stateq(self, idx, self.debug)


                if self.debug: print('[debug] calculating s/n for cell ' + str(idx))

                snr = self.fixset                # Determine snr in cell
                var = 0.
                totphot += self.nphot[idx]

                for ilev in range(self.nlev):
                    self.avepops = (self.pops[ilev, idx] + self.opops[ilev, idx] + self.oopops[ilev, idx])/3.
                    if (self.avepops >= self.minpop):
                        var = np.max([np.abs(self.pops[ilev, idx] - self.avepops)/self.avepops, np.abs(self.opops[ilev, idx] - self.avepops)/self.avepops, np.abs(self.oopops[ilev, idx] - self.avepops)/self.avepops])
                        snr = np.max([snr, var])
                snr = 1./snr
                minsnr = np.min([snr, minsnr])

                if (stage == 1):
                    if (snr >= 1./self.fixset): conv += 1    # Stage 1=FIXSET 
                else:
                    if (snr >= self.goalsnr): conv += 1
                    else:
                        newphot = self.nphot[idx]*2          # Double photons if cell not converged
                        if (newphot >= max_phot):
                            newphot = max_phot
                            exceed += 1
                            print('AMC: *** Limiting nphot in cell ' + str(idx))
                        self.nphot[idx] = newphot

                totphot2 += self.nphot[idx]

            # Report any convergence problems if they occurred

            if (staterr >= 0.): print('### WARNING: stateq did not converge everywhere (err=' + str(staterr))

            if (stage > 1):
                percent = conv/self.ncell*100.
                blowpops(self.outfile, self.molfile, self.snrgoal, minsnr, percent, stage, self.fixset, self.trace) # TODO
                print('AMC: FIXSET fractional error ' + str(1./minsnr) + ', ', + str(percent) + '% converged')
                if (conv == self.ncell):
                    stage = 2
                    print('AMC')
                    print('AMC: FIXSET convergence reached...starting RANDOM')
                    print('AMC:')
                    print('AMC: minimum S/N  |  converged  |     photons  |  increase to')
                    print('AMC: -------------|-------------|--------------|-------------')
                continue                                # Next iteration        

            else:
                if (conv == self.ncell): percent = 100.
                else:
                    if (exceed < self.ncell):
                        percent = conv/self.ncell*100.
                        blowpops(self.outfile, self.molfile, self.snrgoal, minsnr, percent, stage, self.fixset, self.trace) # TODO
                        print('AMC: ' + str(minsnr) + '  |  ' + str(percent) + '% |  ' + str(totphot) + '  |  ' + totphot2)
                        continue                        # Next iteration
                    else:
                        print('### WARNING: Insufficient photons. Not converged.')

            done = True

        # Convergence reached (or bailed out)

        print('AMC: ' + str(minsnr) + '  |  ' + str(percent) + '% |  ' + str(totphot) + '  |  converged')
        print('AMC:')

        #blowpops(self.outfile, self.molfile, self.snrgoal, minsnr, percent, stage, self.fixset, self.trace) # TODO
        print('AMC: Written output to ' + str(self.outfile))
