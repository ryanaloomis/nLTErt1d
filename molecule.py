import numpy as np
from common import *

# converts energy in cm-1 to J
hckb = 100.*hplanck*clight/kboltz

class molecule:
    def __init__(self, sim, molfile, debug):
        self.sim = sim
        self.molfile = molfile

        # Read the file
        try:
            lines = open(molfile).read().splitlines()
        except:
            raise Exception('AMC: molfile cannot be read...abort')

        # Structure of file is known, so start reading the headers
        try:
            self.molname = lines[1].split()[0]
            self.molweight = float(lines[3].split()[0])          
            self.nlev = int(lines[5].split()[0])
            self.nline = int(lines[8 + self.nlev].split()[0])
            self.npart = int(lines[11 + self.nlev + self.nline].split()[0])
            if (self.npart > 2):
                raise Exception('Error: maximum 2 collision partners...abort')

            self.part1id = int(lines[13 + self.nlev + self.nline].split()[0])
            self.ntrans = int(lines[15 + self.nlev + self.nline].split()[0])
            self.ntemp = int(lines[17 + self.nlev + self.nline].split()[0])

            if (self.npart==1):
                self.part2id = None
                self.ntrans2 = 1
                self.ntemp2 = 1
            else:
                self.part2id = int(lines[22 + self.nlev + self.nline + self.ntrans].split()[0])
                self.ntrans2 = int(lines[24 + self.nlev + self.nline + self.ntrans].split()[0])
                self.ntemp2 = int(lines[26 + self.nlev + self.nline + self.ntrans].split()[0])

            #print(self.molname, self.molweight, self.nlev, self.nline, self.npart, self.part1id, self.ntrans, self.ntemp, self.part2id, self.ntrans2, self.ntemp2)

        except:
            raise Exception('AMC: could not read molfile header...abort')



        # Read in the level data
        level_data = lines[7:7+self.nlev]
        for i, line in enumerate(level_data):
            level_data[i] = line.split()
        level_data = np.array(level_data).astype(float)

        # Parse level properties - energies and statistical weights
        self.eterm = level_data[:,1]
        self.gstat = level_data[:,2]


        # Read in the line data
        line_data = lines[10 + self.nlev : 10 + self.nlev+self.nline]
        for i, line in enumerate(line_data):
            line_data[i] = line.split()
        line_data = np.array(line_data).astype(float)

        # Parse line properties - upper/lower levels, frequencies, einstein coefficients
        self.lau = line_data[:,1].astype(int)-1
        self.lal = line_data[:,2].astype(int)-1
        self.aeinst = line_data[:,3]
        self.freq = line_data[:,4]*1.e9
        self.beinstu = self.aeinst*(clight/self.freq)**2/(hplanck*self.freq)/2.
        self.beinstl = np.take(self.gstat, self.lau)/np.take(self.gstat, self.lal)*self.beinstu

        # Add thermal broadening to linewidths
        amass = self.molweight*amu
        for idx in range(sim.ncell):
            sim.model.grid['doppb'][idx] = np.sqrt(sim.model.grid['doppb'][idx]**2. + 2.*kboltz/amass*sim.model.grid['tkin'][idx])

        # Read collision temperatures
        self.coll_temps = np.array((lines[19 + self.nlev + self.nline]).split()).astype(float)

        # Read in the collision data
        self.coll_data = lines[21 + self.nlev + self.nline : 21 + self.nlev + self.nline + self.ntrans]
        for i, line in enumerate(self.coll_data):
            self.coll_data[i] = line.split()
        self.coll_data = np.array(self.coll_data).astype(float)

        # convert cm^3/s to m^3/s
        self.coll_data[:,3:] /= 1.e6


        # Read in second collision partner data if it exists:
        if self.part2id:
            # Read collision temperatures
            self.coll_temps2 = np.array((lines[28 + self.nlev + self.nline + self.ntrans]).split()).astype(float)

            # Read in the collision data
            self.coll_data2 = lines[30 + self.nlev + self.nline + self.ntrans : 30 + self.nlev + self.nline + self.ntrans + self.ntrans2]
            for i, line in enumerate(self.coll_data2):
                self.coll_data2[i] = line.split()
            self.coll_data2 = np.array(self.coll_data2).astype(float)
 
            # convert cm^3/s to m^3/s
            self.coll_data2[:,3:] /= 1.e6


        # Trim off metadata
        self.lcu = self.coll_data[:,1].astype(int) - 1
        self.lcl = self.coll_data[:,2].astype(int) - 1
        self.colld = self.coll_data[:,3:]

        if self.part2id:
            self.lcu2 = self.coll_data2[:,1].astype(int) - 1
            self.lcl2 = self.coll_data2[:,2].astype(int) - 1
            self.colld2 = self.coll_data2[:,3:]


        # Calculate upward/downward rates in all non-empty cells.
        # Interpolate downward rates, but do not extrapolate.
        self.up = np.zeros((self.ntrans, sim.ncell))
        self.down = np.zeros((self.ntrans, sim.ncell))
        
        for idx in range(sim.ncell):
            for t in range(self.ntrans):
                self.down[t,idx] = np.interp(sim.model.grid['tkin'][idx], self.coll_temps, self.colld[t]) 

        self.down[:,sim.model.grid['tkin'] > self.coll_temps[-1]] = self.colld[:,-1, np.newaxis]
        
        for idx in range(sim.ncell):
            for t in range(self.ntrans):
                self.up[t,idx] = self.gstat[self.lcu[t]]/self.gstat[self.lcl[t]]*self.down[t,idx]*np.exp(-hckb*(self.eterm[self.lcu[t]]-self.eterm[self.lcl[t]])/sim.model.grid['tkin'][idx])


        if self.part2id:
            self.up2 = np.zeros((self.ntrans2, sim.ncell))
            self.down2 = np.zeros((self.ntrans2, sim.ncell))
            
            for idx in range(sim.ncell):
                for t in range(self.ntrans2):
                    self.down2[t,idx] = np.interp(sim.model.grid['tkin'][idx], self.coll_temps2, self.colld2[t]) 

            self.down2[:,sim.model.grid['tkin'] > self.coll_temps2[-1]] = self.colld2[:,-1, np.newaxis]
            
            for idx in range(sim.ncell):
                for t in range(self.ntrans2):
                    self.up2[t,idx] = self.gstat[self.lcu2[t]]/self.gstat[self.lcl2[t]]*self.down2[t,idx]*np.exp(-hckb*(self.eterm[self.lcu2[t]]-self.eterm[self.lcl2[t]])/sim.model.grid['tkin'][idx])


        # Initialize some other properties, jbar, etc...
        self.jbar = np.zeros(self.nlev)
