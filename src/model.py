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
            values = model.read_RATRAN(model_file)
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

    @staticmethod
    def read_RATRAN(path, debug=False):
        """
        Read a model file in the RATRAN format. In this format, comments are
        lines starting with '#'. For details of the file format, go to:

            https://personal.sron.nl/~vdtak/ratran/frames.html

        Args:
            path (str): Realtive path to the model file.
            debug (optional[bool]): If True, print debug messages.
        Returns:
            rmax (float): Maximum radius of the model in [m].
            ncell (int): Number of cells in the model.
            tcmb (float): Background (CMB) temperature in [K].
            g2d (float): Gas-to-dust ratio.
            ra (ndarray): Cell inner radius [m].
            rb (ndarray): Cell outer radius [m].
            nH2 (ndarray): Number density of the main collider, usually H2
                [m^-3].
            ne (ndarray): Number density of the secondary collider, usually
                electrons [m^-3].
            nmol (ndarray): Number density of the molecule [m^-3].
            tkin (ndarray): Gas kinetic temperature [K].
            tdust (ndarray): Dust temperature [K].
            telec (ndarray): Electron temperature [K].
            doppb (ndarray): Doppler width of the line [m/s].
            velo (ndarray): [3 x N] shaped array of the radial, vertical and
                azimuthal velocities of the gas in [m/s].
        """

        # Read in the lines and skip the header.
        lines = open(path).read().lower().splitlines()
        header = len(lines)
        lines = [line for line in lines if line[0] != '#']
        header = header - len(lines)

        # Cycle through the keywords and save in dictionary.
        keywords = {}
        for l, line in enumerate(lines):
            if line[0] == '@':
                break
            key, value = line.split('=')
            keywords[key] = value

        # Convert the keywords into appropriate types.
        rmax = float(keywords['rmax'])
        ncell = int(keywords['ncell'])
        tcmb = float(keywords['tcmb'])
        g2d = float(keywords['gas:dust'])
        columns = keywords['columns'].split(',')

        # Load up the data as a numpy array.
        data = np.loadtxt(path, skiprows=l+header+1, dtype=float).T
        if data.shape[0] != len(columns):
            raise ValueError("Wrong number of columns and labels.")

        # Parse the data and convert to correct units.
        ra = data[columns.index('ra')]
        rb = data[columns.index('rb')]
        nH2 = model.get_column('nh', data, columns, 1e10) * 1e6
        ne = model.get_column('ne', data, columns, 0.0) * 1e6
        nmol = model.get_column('nm', data, columns, 1e-2) * 1e6
        tkin = model.get_column('tk', data, columns, 20.0)
        tdust = model.get_column('td', data, columns, 20.0)
        telec = model.get_column('te', data, columns, 0.0)
        doppb = model.get_column('db', data, columns, 1.0) * 1e3
        vr = model.get_column('vr', data, columns, 0.0) * 1e3
        vz = model.get_column('vz', data, columns, 0.0) * 1e3
        va = model.get_column('va', data, columns, 0.0) * 1e3
        velo = np.array([vr, vz, va])

        return (rmax, ncell, tcmb, g2d, ra, rb, nH2, ne, nmol, tkin, tdust,
                telec, doppb, velo)

    @staticmethod
    def get_column(column, data, column_names, fill_value=0.0):
        """Get the data for the specified name, filling with fill_value."""
        try:
            values = data[column_names.index(column)]
        except ValueError:
            values = np.ones(data.shape[1]) * fill_value
        return values
