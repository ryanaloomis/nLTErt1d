# import numpy as np
# from model import model
from simulation import simulation


def amc(source, outfile, molfile, goalsnr, nphot, kappa='jena,thin,e5',
        tnorm=2.735, velocity_function=None, seed=1978, minpop=1e-4,
        fixset=1e-6, debug=False):
    """
    This is the whole simulation.

    Args:
        source (str): Model file.
        outfile (str): File to write population levels to.
        molfile (str): Molecular data file in the LAMDA format.
        goalsnr (float): Goal signal-to-noise ratio of the run.
        nphot (float): Number of photons to use in the radiative transfer.
        kappa (optional[str]): A string decribing the dust parameters. For the
            use of Ossenkopf & Henning (1994) opacities it must take the form:

                kappa_params = 'jena, TYPE, COAG'

            where TYPE must be 'bare', 'thin' or 'thick' and COAG must be 'no',
            'e5', 'e6', 'e7' or 'e8'. Otherwise a power law profile can be
            included. Alternatively, a simple power law can be used where the
            parameters are given by:

                kappa_params = 'powerlaw, freq0, kappa0, beta'

            where freq0 is in [Hz], kappa0 in [cm^2/g], and beta is the
            frequency index. If nothing is given, we assume no opacity.
        tnorm (optional[float]): Background temperature in [K]. Default is the
            CMB at 2.735K.
        velocity_function (optional[str]): Relative path to the file containing
            the function and the function name, for example:

                '~/folders/functions.py, velocity_function'

            to run 'from functions import velocity_function'. If nothing is
            provided, defaults to the velocity provided in the model file.
        seed (optional[int]): Seed for the random number generators.
        minpop (optional[float]): Minimum population for each energy level.
        fixset (optional [float]): The smallest number to be counted.
        debug (optional[bool]): Pring debugging messages.
    """
    print('AMC: ')
    print('AMC: Starting calculations')
    print('AMC:')

    # Set up simulation
    sim = simulation(source=source, outfile=outfile, molfile=molfile,
                     goalsnr=goalsnr, nphot=nphot, kappa=kappa, tnorm=tnorm,
                     velocity_function=velocity_function, seed=seed,
                     minpop=minpop, fixset=fixset, debug=debug)

    # Calculate the level populations
    sim.calc_pops()

if __name__ == "__main__":
    amc('example.mdl', 'example.pop', 'hco+.dat', 20, 1000)
