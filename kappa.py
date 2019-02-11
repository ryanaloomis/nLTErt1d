import numpy as np
from scipy.interpolate import CubicSpline
import scipy.constants as sc


def generate_kappa(kappa_params=None):
    """
    Returns the dust emissivity in [m^2/kg] at a frequency and in cell ID.
    TODO: Make the input directly file paths or floats?

    Args:
        kappa_params (str): A string decribing the dust parameters. For the use
            of Ossenkopf & Henning (1994) opacities it must take the form of:

            kappa_params = 'jena, TYPE, COAG'

            where TYPE must be 'bare', 'thin' or 'thick' and COAG must be 'no',
            'e5', 'e6', 'e7' or 'e8'. Otherwise a power law profile can be
            included. Alternatively, a simple power law can be used where the
            parameters are given by:

            kappa_params = 'powerlaw, freq0, kappa0, beta'

            where freq0 is in [Hz], kappa0 in [cm^2/g], and beta is the
            frequency index. If nothing is given, we assume no opacity.

    Returns:
        kappa: A callable function returning the dust emissivity in [m^2/kg].

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
