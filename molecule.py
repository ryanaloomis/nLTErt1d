import numpy as np
from reading import read_LAMDA
from scipy.interpolate import interp1d
import scipy.constants as sc


class molecule:

    def __init__(self, sim, molfile):
        """
        Initialize the molecule class.

        Args:
            molfile (str): Relative path to the molecular data file.
        """

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

        self.find_blends(sim)


    def find_blends(self, sim):
        """
        Find all line blends and record them
        """
        
        self.blends = []
        for iline in range(self.nline):
            for jline in range(self.nline):
                deltav = (self.freq[jline] - self.freq[iline])*sc.c/self.freq[iline]
                if (np.abs(deltav) < 1.e3) and (iline != jline): #1 km/s is limit for blend #TODO reconsider this value
                    self.blends.append([iline, jline, deltav])
        self.blends = np.array(self.blends)
        
        if (self.blends.shape[0] > 0) and (sim.blending == False):
            print("WARNING: Blends found but line blending is turned off. You may want to turn it on.")

        if self.blends.shape[0] == 0:
            if sim.blending == True:
                print("Blending was turned on but no blends found. Turning blending off.")
            self.blends = np.array([np.zeros(3)]) # doing this to keep the numba compiler happy
            sim.blending = False


    def set_rates(self, Tkin):
        """
        Calculate the upward and downward rates in all empty cells. Interpolate
        all the downward rates but do not extrapolate.

        Args:
            Tkin (ndarray): The gas temperatures in each of the cells.

        Returns:
            up1 (ndarray): Upward rates in each cell from first partner.
            down1 (ndarray): Downward rates in each cell from first partner.
            up2 (ndarray): Upward rates in each cell from second partner.
            down2 (ndarray): Downward rates in each cell from second partner.
        """

        # First collision partner.
        dE = self.eterm[self.lcu] - self.eterm[self.lcl]
        try:
            down1 = interp1d(self.coll_temps, self.colld, fill_value="extrapolate")(Tkin)
        except:
            print("Warning: model grid includes cells with higher temperature than collisional data. Attempting to extrapolate and proceed.")
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
            try:
                down2 = interp1d(self.coll_temps2, self.colld2, fill_value="extrapolate")(Tkin)
            except:
                print("Warning: model grid includes cells with higher temperature than collisional data. Attempting to extrapolate and proceed.")
            up2 = 100. * sc.h * sc.c * dE2[:, None] / sc.k / Tkin[None, :]
            up2 = np.exp(-up2)
            up2 *= (self.gstat[self.lcu2] / self.gstat[self.lcl2])[:, None]
            up2 *= down2

        self.up = up1
        self.down = down1
        self.up2 = up2
        self.down2 = down2
