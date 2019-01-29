import numpy as np
from common import *
from simulation import *

# Returns dust emissivity in m2/kg_dust at frequency freq and in cell id.
# Ossenkopf & Henning Jena models.
#
# Useage in amc.inp:  kappa=jena,TYPE,COAG
# where TYPE = bare / thin / thick
# and   COAG = no / e5 / e6 / e7 / e8

def generate_kappa(kappa_params):
    if kappa_params:
        params = kappa_params.split(',')

    else:
        # default case, set kappa to 0
        def kappa(idx, freq):
            return 0.
        return kappa
    

    if params[0] == 'jena':
        filename = "kappa/jena_" + params[1] + "_" + params[2] + ".tab"
        table = np.loadtxt(filename)
        lamtab = table[:,0]/1.e6
        kaptab = table[:,1]

        def kappa(idx, freq):
            lam_lookup = clight/freq
            kap_interp = np.interp(np.log10(lam_lookup), np.log10(lamtab), kaptab)
            # ...in m2/kg_dust: (0.1 converts cm2/g_dust to m2/kg_dust)
            return kap_interp*0.1
        return kappa

    elif params[0] == 'powerlaw':
        freq0 = float(params[1])
        kappa0 = float(params[2])
        beta = float(params[3])

        def kappa(idx, freq):
            # Simple power law behavior; kappa=powerlaw,freq0,kappa0,beta where freq0 is in Hz, kappa0 in cm2/g_dust, and beta is freq.index.
            # ...in m2/kg_dust: (0.1 converts cm2/g to m2/kg)
            return 0.1*kappa0*(freq/freq0)**beta
        return kappa

    else:
        raise Exception("ERROR: Please supply valid kappa parameters.")

'''

c     Read table on first call; lamtab(i) might be 0, but only for one i
      if ((lamtab(1).eq.0).and.(lamtab(2).eq.0)) then
        open(22,file='ratranjena.tab',status='old')
        do i=1,n
          read(22,*) lamtab(i),kaptab(i)
          lamtab(i)=dlog10(lamtab(i)/1.d6)
          kaptab(i)=dlog10(kaptab(i))
        enddo
        close(22)
      endif

c     Interpolate/extrapolate table (in log units)

      loglam=dlog10(clight/nu)
      call locate(lamtab,n,loglam,j)
      k=min(max(j-(m-1)/2,1),n+1-m)
      call polint(lamtab(k),kaptab(k),m,loglam,logkap,dummy)

c     ...in m2/kg_dust: (0.1 converts cm2/g_dust to m2/kg_dust)
      kappa=0.1*10.**logkap


      RETURN
      END
'''
