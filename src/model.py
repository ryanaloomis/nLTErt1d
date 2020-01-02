import numpy as np


class model:
    """
    The 1D model grid used for the radiative transfer.

    Args:
        model_file (str): Relative path to the model input file.
        model_type (optional[str]): Type of model format. Allowed values
            are currently only 'ratran'.
    """

    def __init__(self, model_file, model_type='ratran'):

        # Read in the model type.
        model_type = model_type.lower()
        if model_type == 'ratran':
            from reading import read_RATRAN
            values = read_RATRAN(model_file)
        else:
            raise ValueError("`model_type` must be: 'ratran'.")

        # Assign the model variables.
        self.rmax = values[0]
        self.ncell = values[1]
        self.tcmb = values[2]
        self.g2d = values[3]
        self.ra = values[4]
        self.rb = values[5]
        self.nh2 = values[6]
        self.ne = values[7]
        self.nmol = values[8]
        self.tkin = values[9]
        self.tdust = values[10]
        self.telec = values[11]
        self.doppb = values[12]
        self.velocities = values[13]

        # Sanity check on variables.
        if self.ncell == 0.0:
            raise ValueError("Zero cells in model.")
        if self.tcmb < 0.0:
            raise ValueError("Negative CMB temperature.")
        if self.velocities.ndim != 2:
            raise ValueError("Unknown velocity format.")

    def velo(self, idx, x=None):
        """Return the (v_r, v_z, v_a) tuple."""
        return self.velocities[:, idx]

    def velo_old(self, idx, x):
        v = np.zeros(3)
        if 'vr' in self.grid:
            v[0] = self.grid['vr'][idx]
        if 'vz' in self.grid:
            v[1] = self.grid['vz'][idx]
        if 'va' in self.grid:
            v[2] = self.grid['va'][idx]
        return v
