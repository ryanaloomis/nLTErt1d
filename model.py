import numpy as np
from common import *

class model:
    def __init__(self, modelfile, debug):
        needed_cols = ['id', 'ra', 'rb', 'nh', 'tk', 'nm', 'db']
        entered_grid = False

        self.rmax = 0                    # Initial settings
        self.zmax = 0
        self.ncell = 0
        self.tcmb = 2.735
        self.gas2dust = 100.
        columns = []
        tmp_grid = []

        lines = open(modelfile).read().splitlines()

        for line in lines:
            if (entered_grid == False):
                if line[0] == '#': continue
                if line[0] == '@': 
                    entered_grid = True
                    continue

                if debug: print('[debug] read header')

                if '=' in line:
                    try:                                # Search keywords (case sensitive)
                        keyword = line.split('=')[0]
                        if (keyword == 'rmax'):
                            self.rmax = float(line.split('=')[1])
                        elif (keyword == 'zmax'):
                            self.zmax = float(line.split('=')[1])
                        elif (keyword == 'ncell'):
                            self.ncell = int(line.split('=')[1])
                        elif (keyword == 'tcmb'):
                            self.tcmb = float(line.split('=')[1])
                        elif (keyword == 'gas:dust'):
                            self.gas2dust = float(line.split('=')[1])
                        elif (keyword == 'columns'):
                            columns = line.split('=')[1].split(',')
                        else:
                            print('AMC: cannot understand input')
                            print(line)
                            print('AMC: skipping...')
                        continue

                    except:
                        print('AMC: cannot understand input')
                        print(line)
                        print('AMC: skipping...')
                        continue

                # End of header reached: check on validity
                
                if (rmax <= 0.):
                    raise ValueError('AMC: <rmax> missing or 0...abort')
                if (ncell <= 0.):
                    raise ValueError('AMC: <ncell> missing or 0...abort')
                if (len(columns) <= 0):
                    raise ValueError('AMC: <columns> must be defined...abort')

                # Check for missing columns
                if not(all(col in columns  for col in needed_cols)):
                    raise ValueError('AMC: a column is missing...abort')

                if (zmax>0):
                    if ('za' not in columns) or (zb not in columns):
                        raise ValueError('AMC: a column is missing...abort')

                if debug: print('[debug] validated header')


            else:
                # Read columns into grid
                tmp_grid.append(line.split())

        try:
            if len(tmp_grid[0]) != len(columns):
                raise ValueError('AMC: grid is not correctly sized to match # of columns...abort')
            tmp_grid = np.array(tmp_grid, dtype='float')
            self.grid = dict(zip(columns, tmp_grid.T))
        except:
            raise Exception('AMC: invalid value in grid...abort')

        # Convert units and dict keywords where appropriate
        self.grid['nh2'] = self.grid.pop('nh')*1.e6         # [cm^-3] -> [m^-3]
        self.grid['nmol'] = self.grid.pop('nm')*1.e6        # [cm^-3] -> [m^-3]
        self.grid['tkin'] = self.grid.pop('tk')             # [K]
        self.grid['doppb'] = self.grid.pop('db')*1.e3       # [km/s] -> [m/s]

        if 'td' in columns:
            self.grid['tdust'] = self.grid.pop('td')        # [K]

        if 'ne' in columns:
            self.grid['ne'] = self.grid.pop('ne')*1.e6      # [cm^-3] -> [m^-3]

        if 'vr' in columns:
            self.grid['vr'] = self.grid.pop('vr')*1.e3      # [km/s] -> [m/s]

        if 'vz' in columns:
            self.grid['vz'] = self.grid.pop('vz')*1.e3      # [km/s] -> [m/s]

        if 'va' in columns:
            self.grid['va'] = self.grid.pop('va')*1.e3      # [km/s] -> [m/s]




    def velo(self, idx, x):
        v = np.zeros(3)
        v[0] = self.grid['vr'][idx]
        v[1] = self.grid['vz'][idx]
        v[2] = self.grid['va'][idx]

        return v


