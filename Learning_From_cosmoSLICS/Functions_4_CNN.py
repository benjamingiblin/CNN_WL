# 20/01/2020
# Functions for use in the CNN code.

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

import matplotlib.gridspec as gridspec
from matplotlib import rcParams
import matplotlib.pyplot as plt
# Some font setting                                                                                                                            
rcParams['ps.useafm'] = True
rcParams['pdf.use14corefonts'] = True
font = {'family' : 'serif',
        'weight' : 'normal',
        'size'   : 14} # 19                                                                                                                    
plt.rc('font', **font)


def Slow_Read(CS_DIR, mock_Type, Cosmols_Raw, mockIDs, Train_mockIDs, Test_mockIDs,
              Realisations, ZBlabel, Res, Test_Realisation_num, Noise):
    
    # Start preparing the arrays to store the shear maps & corresponding cosmological parameters:
    Shear = np.zeros([ len(mockIDs)*2*Realisations*4, 2*len(ZBlabel), Res,Res ])
                                                              # This has the dimensions of 
                                                              # [num_maps, num_channels, pxl, pxl ]
                                                              # Here num_maps=26(cosmols)*2(seeds)*50(realisations)*4quadrants
                                                              # And num_channels (previously 3 for colours) is
                                                              # 2 for shear components * number of z-bins.

    Test_Shear = np.zeros([ len(Test_mockIDs)*2*Test_Realisation_num*4, 2*len(ZBlabel), Res,Res ])
    Train_Shear = np.zeros([ len(Test_mockIDs)*2*(Realisations-Test_Realisation_num)*4 + len(Train_mockIDs)*2*Realisations*4,
                         2*len(ZBlabel), Res,Res ])

    # Create arrays that will match the cosmol params to the shear images
    Cosmols = np.zeros([ Shear.shape[0], 4 ])              # store all cosmols
    Test_Cosmols = np.zeros([ Test_Shear.shape[0], 4 ])    # store the test cosmols
    Train_Cosmols = np.zeros([ Train_Shear.shape[0], 4 ])  # & the training cosmols
    # These two arrays aren't used in the code. They're just for easily checking
    # the mockID that each row of Test & Train correspond to:
    Test_Cosmols_IDs = []
    Train_Cosmols_IDs = []

    # ...Now that the arrays are all prepared, begin cycling through the data,
    # reading them in and sorting into either either Train or Test maps/cosmologies:

    t1 = time.time()                                    # Start the timer
    map_count=0                                         # counter for scrolling through map realisations (cosmol, seeds, los & quads)
    map_count_test=0
    map_count_train=0
    for i in range(len(mockIDs)):                    # Scroll through the mock IDs
        for r in range(Realisations):                # Scroll through the 25 realisations
            for seed in ['a', 'f']:                  # Scroll through the 2 seeds 
                for q in range(4):                   # Scroll through the 4 quadrants
            
                    channel_count = 0                           # count for cycling through channels (shear components & zbins)
                    channel_count_test = 0
                    channel_count_train = 0
            
                    for zb in range(len(ZBlabel)):              # scroll through z-bins
                        for s in range(2):                      # Scroll through 2 shear components

                            # Open the map
                            f = fits.open(CS_DIR+'%s_%s/%s/Shear_Maps/Res%sx%s/%s/Shear%s_LOS%s_SN%s_Quad%s.fits'
                                    %(mockIDs[i],seed,mock_Type,Res,Res,ZBlabel[zb],s+1,r+1,Noise,q))
                        
                            Shear[map_count,channel_count,:,:] = f[0].data # store shear
                            Cosmols[map_count,:] = Cosmols_Raw[i,:]        # and corresponding cosmols

                            # Decide if this mockID & realisation corresponds to Test or Train:
                            if (mockIDs[i] in Test_mockIDs) and r<Test_Realisation_num:
                                # Store it in the test set
                                Test_Shear[map_count_test,channel_count_test, :, :] = f[0].data
                                Test_Cosmols[map_count_test,:] = Cosmols_Raw[i,:]
                                channel_count_test+=1
                                if channel_count_test == len(ZBlabel)*2:
                                    # We have finished reading in all channels, so also increment the map number:
                                    map_count_test+=1
                                    # and store the Test_Cosmol_ID 
                                    Test_Cosmols_IDs.append( mockIDs[i] )
                                
                            else:
                                # store it in the training set
                                Train_Shear[map_count_train,channel_count_train, :, :] = f[0].data
                                Train_Cosmols[map_count_train,:] = Cosmols_Raw[i,:]
                                channel_count_train+=1
                                if channel_count_train == len(ZBlabel)*2:
                                    # We have finished reading in all channels, so also increment the map number:
                                    map_count_train+=1
                                    # and store the Test_Cosmol_ID
                                    Train_Cosmols_IDs.append( mockIDs[i] )
                            
                            f.close()
                            channel_count+=1
                    map_count+=1                           # Increment map count
    t2 = time.time()                                       # Take another timer measurement
    print("Time taken to read in maps is %.1f seconds." %(t2-t1))

    return Shear, Train_Shear, Test_Shear, Cosmols, Train_Cosmols, Test_Cosmols, Train_Cosmols_IDs, Test_Cosmols_IDs


def Transform_Data(data, mean, std):
    new_data = np.zeros_like( data)
    for i in range( len(mean) ):
        new_data[:,i,:,:] = (data[:,i,:,:] - mean[i]) / std[i]
    return new_data

def Untransform_Data(data, mean, std):
    new_data = np.zeros_like( data)
    for i in range( len(mean) ):
        new_data[:,i,:,:] = std[i]*data[:,i,:,:] + mean[i]
    return new_data

    
def Plot_Accuracy(pred,test):
    # first avg & std the predictions for each test cosmology
    # Find the unique values of Omega_m, to figure out how many test cosmologies there are
    indices = np.unique( test[:,0], return_index=True )[1]
    test_unique = test[ np.sort(indices), : ] # the unique test cosmologies
    
    pred_avg = np.zeros_like( test_unique ) 
    pred_std = np.zeros_like( test_unique )

    for i in range( test_unique.shape[0] ):     # scroll through the unique cosmologies
        # Get the indices of rows that match the 1st unique cosmology
        # these are the predictions we need to avg and take the std of.
        idx = np.where( test[:,0] == test_unique[i,0] )[0]
        for j in range( test.shape[1] ):       # scroll through (Omega_m, S_8, h, w0)
            pred_avg[i,j] = np.mean( pred[idx,j] )
            pred_std[i,j] = np.std( pred[idx,j] )

    # Now plot the accuracies of the predictions
    fig = plt.figure(figsize = (8,8))
    if pred.shape[1] % 2 ==0:
        # an even number of panels to produce
        nrows = int(pred.shape[1]/2)
    else:
        # odd number of panels, round up number of rows
        nrows = int(pred.shape[1]/2) + 1
        
    gs1 = gridspec.GridSpec(2, nrows )
    colors = [ 'magenta', 'darkblue', 'dimgrey', 'orange']
    plot_labels = [ r'$\Omega_{\rm m}$', r'$S_8$', r'$h$', r'$w_0$' ]
    for j in range( test.shape[1] ):           # scroll through (Omega_m, S_8, h, w0) 
        ax = plt.subplot(gs1[j], adjustable='box')
        ax.errorbar( np.arange( test_unique.shape[0] )+1,
                     100.*(pred_avg[:,j]/test_unique[:,j] -1.),
                     yerr=100.*( pred_std[:,j]/test_unique[:,j] /np.sqrt( len(test_unique[:,j]) ) ),
                     color=colors[j], fmt='o')
        ax.set_ylabel(r'%s Accuracy' %plot_labels[j] + r' [\%]')
    plt.subplots_adjust(hspace=0)
    plt.show()
    return


