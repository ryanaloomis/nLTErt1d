import numpy as np
from reading import read_LAMDA
from scipy.interpolate import interp1d
import scipy.constants as sc


class molecule:
    """
    Molecular LAMDA datafile class.

    Args:
        sim (simulation class): ?
        molfile (str): ?
    """

    def __init__(self, sim, molfile):
        input = read_LAMDA(molfile)

        self.molename, self.molweight, self.nlev, self.nline = input[:4]
        self.npart, self.level_idx, self.eterm, self.gstat = input[4:8]
        self.lau, self.lal, self.aeinst, self.freq, self.beinstu = input[8:13]
        self.beinstl, part1, part2 = input[13:]

        self.part1id, self.ntrans, self.ntemp, self.coll_temps = part1[:4]
        self.part2id, self.ntrans2, self.ntemp2, self.coll_temps2 = part2[:4]
        self.lcu, self.lcl, self.colld = part1[4:]
        self.lcu2, self.lcl2, self.colld2 = part2[4:]

        self.up, self.down = None, None
        self.up2, self.down2 = None, None

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

        # Return.
        return up1, down1, up2, down2

    def set_rates(self, Tkin, extrapolate=True):
        """Set the collision rates."""
        up1, down1, up2, down2 = self.get_rates(Tkin, extrapolate)
        self.up, self.down, self.up2, self.down2 = up1, down1, up2, down2
