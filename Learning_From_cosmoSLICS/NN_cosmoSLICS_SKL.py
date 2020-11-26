# 14/01/2020, B. M. Giblin, Postdoc, Edinburgh
# Build a simple(?) neural net to train on cosmoSLICS maps

from sklearn.neural_network import MLPRegressor       # Neural net module
from sklearn.preprocessing import StandardScaler      # Used to normalise training and test data (helps speed up net)
from sklearn.model_selection import train_test_split  # Used to easily split data into a training and a test sample
                                      

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
Shear = np.zeros([ 2,len(mockIDs),2*Realisations,Res*Res ])
#Shear2 = np.zeros_like(Shear1)

t1 = time.time()                                 # Start the timer
for s in range(2):                               # Scroll through 2 shear components

    for i in range(len(mockIDs)):                # Scroll through the mock IDs
        k=0                                      # counter for storing maps
        for seed in ['a', 'f']:                  # Scroll through the 2 seeds
            for r in range(Realisations):        # Scroll through the 25 realisations
                f = fits.open(CS_DIR+'%s_%s/GalCat/Shear_Maps/Res%sx%s/%s/Shear%s_LOS%s_SN%s.fits'
                                    %(mockIDs[i],seed,Res,Res,ZBlabel,s+1,r+1,Noise))
                Shear[s,i,k,:] = np.ndarray.flatten(f[0].data)      # Flatten the 2D input map
                f.close()
                k+=1                           # Increase counter
                
t2 = time.time()                               # Take another timer measurement
print("Time taken to read in maps is %.1f seconds." %(t2-t1))


# Read in the corresponding cosmological parameters 
# These will be the output of the neural net!
Cosmols = np.loadtxt(CS_DIR+'/cosmoSLICS_Cosmologies_Omm-S8-h-w0-sig8-sig8bf.dat')[:,0:4] # Only take the frst 4 columns





# NEURAL NET STUFF - SEE THE FOLLOWING FOR RESOURCES.
# https://scikit-learn.org/stable/modules/neural_networks_supervised.html
# https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPClassifier.html#sklearn.neural_network.MLPClassifier
# NOTES TO SELF: Fluri et al. use:
# Fluri et al. use solver='adam', activation='relu'
# len(hidden_layer_size) is the number of hidden layers.
# The elements of hidden_layer_size are the number of nodes in each layer...
# ...best to have these somewhere between the input size (Res*Res) and the output size (4)

# Scale the maps to have mean~0 and standard deviation~1 (makes neural net more accurate)
scaler = StandardScaler()
Shear_rs = np.copy( Shear[0,:,0,:] ) # Just extracting shear1, realisation 1
scaler.fit(Shear_rs)
Shear_rs = scaler.transform(Shear_rs)

# Split the data into a training and a test sample
# see https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html
X_train, X_test, Y_train, Y_test = train_test_split(Shear_rs, Cosmols, test_size=0.2  )


t1 = time.time()
# Set the architecture of the neural net
mlp = MLPRegressor(solver='adam', activation='relu', hidden_layer_sizes=(8192),
                    learning_rate='constant',
                    max_iter=10000,
                    random_state=1)


mlp.fit(X_train, Y_train) # Train the neural net
t2 = time.time()
print("Time taken to train neural net is %.1f seconds." %(t2-t1))
print("Loss (should be decreasing) is ", mlp.loss_)


# Make predictions
predictions = mlp.predict(X_test)



    
