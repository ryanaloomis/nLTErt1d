"""Functions to read and write the necessary files."""

import numpy as np
import scipy.constants as sc


def read_LAMDA(path):
    """
    Read a LAMDA molecular data file.

    Args:
        path (str): Relative path to the LAMDA data file.

    Returns:
        name (str): Name of the molecule.
        nu (float): Molecular weight of the molecule.
        nlev (int): Number of energy levels.
        nlines (int): Number of transitions.
        npart (int): Number of collision partners.
        level_idx (ndarray): Energy level index.
        eterm (ndarray): Energy of each level [cm^-1].
        gstat (ndarray): Statistical weight of each level.
        lau (ndarray): LAMDA upper level index.
        lal (ndarray): LAMDA lower level index.
        Aul (ndarray): Einstein A coefficients [s^-1]
        freq (ndarray): Frequency of the transitions [Hz].
        Bu (ndarray): Einstein B coefficient for emission [s^-1].
        Bl (ndarray): Einstein B coefficient for absoption [s^-1].
        part1 (list): A list containing the partner ID, the number of
            collisional levels, the number of collisional temperatures and
            collisional temperatures, the LAMDA upper and lower level indices
            and the the collisional rates.
        part2 (list): A list containing the partner ID, the number of
            collisional levels, the number of collisional temperatures and
            collisional temperatures, the LAMDA upper and lower level indices
            and the the collisional rates.
    """

    # Read in the lines.
    lines = open(path).read().lower().splitlines()

    # Basic properties of the file.
    name = lines[1]
    mu = float(lines[3])
    nlev = int(lines[5])
    nlines = int(lines[6 + nlev + 2])
    npart = int(lines[6 + nlev + nlines + 5])
    if npart > 2:
        raise Exception("Maximum 2 collision partners only.")

    # Read in the energy levels and split into level number,
    # energy (cm^-1) and statistical weight.
    # TODO: Check this is OK for non-linear rotators.
    levels = [l.split() for l in lines[7:7+nlev]]
    levels = np.array(levels).astype('float').T
    if levels.shape[1] != nlev:
        raise Exception("Unexpected number of energy levels.")
    level_idx, eterm, gstat = levels[:3]

    # Read in the transitions.
    transitions = [l.split() for l in lines[10+nlev:10+nlev+nlines]]
    transitions = np.array(transitions).astype('float').T
    if transitions.shape[1] != nlines:
        raise Exception("Unexpected number of transitions.")
    lau, lal = transitions[1:3].astype('int') - 1
    Aul = transitions[3]
    freq = transitions[4] * 1e9
    Bu = Aul * sc.c**2 / freq**3 / 2. / sc.h
    Bl = Bu * np.take(gstat, lau) / np.take(gstat, lal)

    # Get the collisional data and rates for the first partner.
    part1_id = int(lines[13 + nlev + nlines].split()[0])
    part1_ntrans = int(lines[15 + nlev + nlines])
    part1_ntemp = int(lines[17 + nlev + nlines])

    idx = 19 + nlev + nlines
    part1_ctemps = np.array(lines[idx].split()).astype('float')
    idx = 21 + nlev + nlines
    part1_crates = [l.split() for l in lines[idx:idx+part1_ntrans]]
    part1_crates = np.array(part1_crates).astype('float')
    if part1_crates.shape != (part1_ntrans, part1_ntemp + 3):
        raise Exception('Unexpected number of collision rates.')
    part1_lcu, part1_lcl = part1_crates[:, 1:3].T.astype('int') - 1
    part1_crates = part1_crates[:, 3:] / 1e6
    part1 = [part1_id, part1_ntrans, part1_ntemp, part1_ctemps,
             part1_lcu, part1_lcl, part1_crates]

    # Get the collision rates for the second partner.
    if npart == 1:
        part2_id = None
        part2_ntrans = 1
        part2_ntemp = 1
        part2_ctemps = np.zeros(1)
        part2_lcu = np.zeros(1).astype('int')
        part2_lcl = np.zeros(1).astype('int')
        part2_crates = np.zeros(1)

    else:
        idx = nlev + nlines + part1_ntrans
        part2_id = int(lines[22 + idx].split()[0])
        part2_ntrans = int(lines[24 + idx])
        part2_ntemp = int(lines[26 + idx])
        part2_ctemps = np.array(lines[28 + idx].split()).astype('float')
        part2_crates = [l.split() for l in lines[30+idx:30+idx+part2_ntrans]]
        part2_crates = np.array(part2_crates).astype('float')
        if part2_crates.shape != (part2_ntrans, part2_ntemp + 3):
            raise Exception('Unexpected number of collision rates.')
        part2_lcu, part2_lcl = part2_crates[1:3].astype('int') - 1
        part2_crates = part2_crates[:, 3:] / 1e6
    part2 = [part2_id, part2_ntrans, part2_ntemp, part2_ctemps,
             part2_lcu, part2_lcl, part2_crates]

    return (name, mu, nlev, nlines, npart, level_idx, eterm, gstat, lau, lal,
            Aul, freq, Bu, Bl, part1, part2)


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
    nH2 = get_column('nh', data, columns, 1e10) * 1e6
    ne = get_column('ne', data, columns, 0.0) * 1e6
    nmol = get_column('nm', data, columns, 1e-2) * 1e6
    tkin = get_column('tk', data, columns, 20.0)
    tdust = get_column('td', data, columns, 20.0)
    telec = get_column('te', data, columns, 0.0)
    doppb = get_column('db', data, columns, 1.0) * 1e3
    vr = get_column('vr', data, columns, 0.0) * 1e3
    vz = get_column('vz', data, columns, 0.0) * 1e3
    va = get_column('va', data, columns, 0.0) * 1e3
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
