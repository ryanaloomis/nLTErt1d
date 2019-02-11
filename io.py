"""Functions to read and write the necessary files."""

import numpy as np


def read_RATRAN(path, debug=False):
    """
    Read a model file in the RATRAN format. In this format, comments are lines
    starting with '#'. For details of the file format, go to:

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
        nH2 (ndarray): Number density of the main collider, usually H2 [m^-3].
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
    nH2 = get_column('nh', data, keywords, 1e10) * 1e6
    ne = get_column('ne', data, keywords, 0.0) * 1e6
    nmol = get_column('nmol', data, keywords, 1e-2) * 1e6
    tkin = get_column('tkin', data, keywords, 20.0)
    tdust = get_column('dust', data, keywords, 20.0)
    telec = get_column('te', data, keywords, 0.0)
    doppb = get_column('db', data, keywords, 1.0) * 1e3
    vr = get_column('vr', data, keywords, 0.0) * 1e3
    vz = get_column('vz', data, keywords, 0.0) * 1e3
    va = get_column('va', data, keywords, 0.0) * 1e3
    velo = np.array([vr, vz, va])

    return (rmax, ncell, tcmb, g2d, ra, rb, nH2, ne, nmol, tkin, tdust,
            telec, doppb, velo)


def get_column(column, data, column_names, fill_value=0.0):
    """Get the data for the specified name. If not there, use fill_value."""
    try:
        values = data[column_names.index(column)]
    except:
        values = np.ones(data.shape[1]) * fill_value
    return values
