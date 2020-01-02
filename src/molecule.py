import numpy as np
from scipy.interpolate import interp1d
import scipy.constants as sc


class molecule:
    """
    Reads in a molecular datafile.

    Has some small functionality to interpolate collision rates for a given
    temperature and to detect any blended transitions.

    Args:
        molfile (str): Path to the molecular datafile.
        file_type (optional[str]): The type of file with the molecular data in.
            Currently only accepts `'LAMDA'` formats.
    """

    def __init__(self, molfile, file_type='LAMDA'):

        # Read a specific filetype.
        file_type = file_type.lower()
        if file_type.lower() == 'lamda':
            input = molecule.read_LAMDA(molfile)
        else:
            raise ValueError("`file_type` must be: 'LAMDA'.")

        # Parse all the input values.
        self.molename, self.molweight, self.nlev, self.nline = input[:4]
        self.npart, self.level_idx, self.eterm, self.gstat = input[4:8]
        self.lau, self.lal, self.aeinst, self.freq, self.beinstu = input[8:13]
        self.beinstl, part1, part2 = input[13:]
        self.part1id, self.ntrans, self.ntemp, self.coll_temps = part1[:4]
        self.part2id, self.ntrans2, self.ntemp2, self.coll_temps2 = part2[:4]
        self.lcu, self.lcl, self.colld = part1[4:]
        self.lcu2, self.lcl2, self.colld2 = part2[4:]

    def get_blends(self, blend_limit=1.0e3):
        """
        Find all line blends and record them.

        Args:
            blend_limit (optional[float]): Difference between line centres in
                [m/s] which are considered to be blended.

        Returns:
            An array of the blended components with a shape of `(nblends, 3)`,
            with each blend having the two line indices and the velocity
            separation between them in [m/s].
        """
        blends = []
        for iline in range(self.nline):
            # CHECK: This will duplicate i > j with j > i. Is this necessary?
            for jline in range(self.nline):
                if iline != jline:
                    deltav = (self.freq[jline] - self.freq[iline])
                    deltav *= sc.c / self.freq[iline]
                    if (np.abs(deltav) < blend_limit):
                        blends += [[iline, jline, deltav]]
        return np.array(blends)

    def get_rates(self, Tkin, fill_value='extrapolate'):
        """
        Calculate the upward and downward rates in all empty cells. Interpolate
        all the downward rates but do not extrapolate.

        Args:
            Tkin (ndarray): The gas temperatures in each of the cells.
            fill_value (optional): To be passed to scipy.interp1d. If
                `'extrapolate'`, the default, will extrapolate the provided
                collision rates.

        Returns:
            up1 (ndarray): Upward rates in each cell from first partner.
            down1 (ndarray): Downward rates in each cell from first partner.
            up2 (ndarray): Upward rates in each cell from second partner.
            down2 (ndarray): Downward rates in each cell from second partner.
        """

        # First collision partner.
        dE = self.eterm[self.lcu] - self.eterm[self.lcl]
        down1 = interp1d(self.coll_temps, self.colld,
                         fill_value=fill_value)(Tkin)
        up1 = 100. * sc.h * sc.c * dE[:, None] / sc.k / Tkin[None, :]
        up1 = np.exp(-up1)
        up1 *= (self.gstat[self.lcu] / self.gstat[self.lcl])[:, None]
        up1 *= down1

        # Second collision partner.
        if self.npart == 1:
            down2 = np.zeros(down1.shape)
            up2 = np.zeros(up1.shape)
        else:
            dE2 = self.eterm[self.lcu2] - self.eterm[self.lcl2]
            down2 = interp1d(self.coll_temps2, self.colld2,
                             fill_value=fill_value)(Tkin)
            up2 = 100. * sc.h * sc.c * dE2[:, None] / sc.k / Tkin[None, :]
            up2 = np.exp(-up2)
            up2 *= (self.gstat[self.lcu2] / self.gstat[self.lcl2])[:, None]
            up2 *= down2

        return up1, down1, up2, down2

    def get_lines(self, freqax):
        """Returns the indices of lines which fall in the frequency axis"""
        fmin, fmax = freqax.min(), freqax.max()
        lines = [l for l, f in enumerate(self.freq) if (f > fmin) & (f < fmax)]
        return np.squeze(lines)

    @staticmethod
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
                collisional temperatures, the LAMDA upper and lower level
                indices and the the collisional rates.
            part2 (list): A list containing the partner ID, the number of
                collisional levels, the number of collisional temperatures and
                collisional temperatures, the LAMDA upper and lower level
                indices and the the collisional rates.
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
            part2_lcu = np.ones(1).astype('int')
            part2_lcl = np.zeros(1).astype('int')
            part2_crates = np.zeros(1)

        else:
            idx = nlev + nlines + part1_ntrans
            part2_id = int(lines[22 + idx].split()[0])
            part2_ntrans = int(lines[24 + idx])
            part2_ntemp = int(lines[26 + idx])
            part2_ctemps = np.array(lines[28 + idx].split()).astype('float')
            part2_crates = lines[30+idx:30+idx+part2_ntrans]
            part2_crates = [l.split() for l in part2_crates]
            part2_crates = np.array(part2_crates).astype('float')
            if part2_crates.shape != (part2_ntrans, part2_ntemp + 3):
                raise Exception('Unexpected number of collision rates.')
            part2_lcu, part2_lcl = part2_crates[:, 1:3].T.astype('int') - 1
            part2_crates = part2_crates[:, 3:] / 1e6
        part2 = [part2_id, part2_ntrans, part2_ntemp, part2_ctemps,
                 part2_lcu, part2_lcl, part2_crates]

        # Return all the variables.
        return (name, mu, nlev, nlines, npart, level_idx, eterm, gstat, lau,
                lal, Aul, freq, Bu, Bl, part1, part2)
