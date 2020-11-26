# 20/01/2020
# Simple code to read in cosmoSLICS shear maps and corresponding cosmol. parameters

# DETAILS ABOUT cosmoSLICS:

# 1. There are 26 cosmologies with IDs ['fid', '00', '01', '02',...,'24']

# 2. Each cosmology has two input "seeds" designated 'a' or 'f'
# the seeds determine the initial conditions in the simulated Universe
# And there are two specially chosen such that the average of the power spectra
# in the 'a' and the 'f' simulated Universes is very close to the
# theoretical predictions.

# 3. For each cosmology and seed, there are 25 different realisations of the
# simulated Universe numbered 1 to 25. These give us an idea of the variance
# in the ways a Universe with a given cosmology can evolve (necessary for
# statistical inference of the cosmological parameters in data).

# The cosmoSLICS shear component 1 map, for cosmology 00, seed a, and realisation 1 is stored at:
# /home/bengib/cosmoSLICS/00_a/GalCat/Shear_Maps/Res128x128/ZBcut0.1-1.2/Shear1_LOS1_SNNone.fits

# More info on cosmoSLICS can be found in Harnois-Deraps, Giblin and Joachimi (2019):
# https://arxiv.org/pdf/1905.06454.pdf


import numpy as np                                # Useful for managing/reading data
from astropy.io import fits                       # For reading FITS files
import time                                       # Used to time parts of the code.
import sys                                        # Used to exit code without error, and read in inputs from command line    


ZBlabel = 'ZBcut0.1-1.2' # The redshift range imposed on the maps
Noise = 'None'           # The shape noise level in the maps
Res = 128                # The number of pxls on each side of the maps                                                       

# Assemble a list of the cosmoSLICS 26 IDs: ['fid', '00', '01',...,'24']
mockIDs = ['fid']
for i in range(25):
    mockIDs.append('%.2d' %i)


# Read in the shear maps                                                                                                      
# These are the Input of our neural net!                                                                                      

CS_DIR = '/home/bengib/cosmoSLICS/'          # cosmoSLICS directory                                                           
Realisations = 25
Shear = np.zeros([ 2,len(mockIDs),2*Realisations,Res,Res ])


t1 = time.time()                                 # Start the timer
for s in range(2):                               # Scroll through 2 shear components
    for i in range(len(mockIDs)):                # Scroll through the mock IDs
        k=0                                      # counter for storing maps
        for seed in ['a', 'f']:                  # Scroll through the 2 seeds
            for r in range(Realisations):        # Scroll through the 25 realisations
                f = fits.open(CS_DIR+'%s_%s/GalCat/Shear_Maps/Res%sx%s/%s/Shear%s_LOS%s_SN%s.fits'
                                    %(mockIDs[i],seed,Res,Res,ZBlabel,s+1,r+1,Noise))
                Shear[s,i,k,:] = f[0].data      
                f.close()
                k+=1                           # Increase counter
t2 = time.time()                               # Take another timer measurement
print("Time taken to read in maps is %.1f seconds." %(t2-t1))


# Read in the corresponding cosmological parameters
# These will be the output of the neural net!
Cosmols = np.loadtxt(CS_DIR+'/cosmoSLICS_Cosmologies_Omm-S8-h-w0-sig8-sig8bf.dat')[:,0:4] # Only take the frst 4 columns      


