import numpy as np


class model:

    def __init__(self, model_file, model_type='ratran', debug=False):
        """
        Initialise the model grid.

        Args:
            model_file (str): Relative path to the model input file.
            model_type (optional[str]): Type of model format. Allowed values
                are currently only 'ratran'.
            debug (optional[bool]): If True, pring debug messages.
        """

        model_type = model_type.lower()
        if model_type == 'ratran':
            from reading import read_RATRAN
            values = read_RATRAN(model_file)
        else:
            raise ValueError("`model_type` must be: 'ratran'.")

        self.rmax = values[0]
        self.ncell = values[1]
        self.tcmb = values[2]
        self.gas2dust = values[3]
        self.ra = values[4]
        self.rb = values[5]
        self.nh2 = values[6]
        self.ne = values[7]
        self.nmol = values[8]
        self.tkin = values[9]
        self.tdust = values[10]
        self.telec = values[11]
        self.doppb = values[12]
        self.velo = values[13]

        # -- INCLUDE THE OLDER VERSION STUFF BELOW. -- #
        needed_cols = ['id', 'ra', 'rb', 'nh', 'tk', 'nm', 'db']
        entered_grid = False

        self.rmax = 0                    # Initial settings
        self.zmax = 0
        self.ncell = 0
        self.tcmb = 2.735
        self.gas2dust = 100.
        columns = []
        tmp_grid = []

        lines = open(model_file).read().splitlines()

        for line in lines:
            if not entered_grid:
                if line[0] == '#':
                    continue
                if line[0] == '@':
                    entered_grid = True
                    continue

                if debug:
                    print('[debug] read header')

                # Search keywords (case sensitive)
                if '=' in line:
                    try:
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

                if (self.rmax <= 0.):
                    raise ValueError('AMC: <rmax> missing or 0...abort')
                if (self.ncell <= 0.):
                    raise ValueError('AMC: <ncell> missing or 0...abort')
                if (len(columns) <= 0):
                    raise ValueError('AMC: <columns> must be defined...abort')

                # Check for missing columns
                if not(all(col in columns for col in needed_cols)):
                    raise ValueError('AMC: a column is missing...abort')

                if (self.zmax > 0):
                    if ('za' not in columns) or ('zb' not in columns):
                        raise ValueError('AMC: a column is missing...abort')

                if debug:
                    print('[debug] validated header')

            else:
                # Read columns into grid
                tmp_grid.append(line.split())

        try:
            if len(tmp_grid[0]) != len(columns):
                raise ValueError('AMC: grid is not correctly sized to'
                                 'match # of columns...abort')
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
        if 'vr' in self.grid:
            v[0] = self.grid['vr'][idx]
        if 'vz' in self.grid:
            v[1] = self.grid['vz'][idx]
        if 'va' in self.grid:
            v[2] = self.grid['va'][idx]

        return v
