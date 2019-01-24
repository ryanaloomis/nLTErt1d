import numpy as np

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


        # Read in the line data
        line_data = lines[10 + self.nlev : 10 + self.nlev+self.nline]
        for i, line in enumerate(line_data):
            line_data[i] = line.split()
        line_data = np.array(line_data).astype(float)


        # Read in the collision data
        coll_data = lines[21 + self.nlev + self.nline : 21 + self.nlev + self.nline + self.ntrans]
        for i, line in enumerate(coll_data):
            coll_data[i] = line.split()
        coll_data = np.array(coll_data).astype(float)

        print coll_data





        # Initialize some other properties, jbar, etc...
        self.jbar = np.zeros(self.nlev)
