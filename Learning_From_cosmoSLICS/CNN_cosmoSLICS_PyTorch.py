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
import os

from Functions_4_CNN import Slow_Read, Transform_Data, Untransform_Data, Plot_Accuracy


mock_Type = 'KV450'
ZBlabel = ['ZBcut0.1-0.3', 'ZBcut0.3-0.5', 'ZBcut0.5-0.7', 'ZBcut0.7-0.9', 'ZBcut0.9-1.2']
          #['ZBcut0.1-1.2'] # The redshift range imposed on the maps
Noise = 'None'           # The shape noise level in the maps
Res = 128                # The number of pxls on each side of the maps

Fast_Or_Slow_Read = "Fast"    # if "Slow", reads all shear maps one-by-one into array
                              # (always run "Slow" first time).
                              # If "Fast", reads a pickled version of the data
                              # Can also set to None if running multiple times in ipython.
                           
Split_Type = "Fid"            # "Fid" meaning fiducial, means the 6 cosmols closest to the middle
                              # of the parameter space are the test cosmologies, with the first 12
                              # LOS per seed being test maps, with the final 13 per seed being training maps.
                           
if Split_Type == "Fid":
    Test_mockIDs = ['00','05','07','12','18','fid']     # the test mock IDs
    Test_Realisation_num = 12 # Use this many of the lines of sight in Test set
                              # So the other (Realisations-Test_Realisation_num) will be in the training set  

else:
    print("The only Split_Type currently supported is Fid. Please set accordingly. EXITING.")
    sys.exit()

                              
# Assemble a list of the cosmoSLICS 26 IDs: ['fid', '00', '01',...,'24']
mockIDs = []          # store all mockIDS
Train_mockIDs = []    # store the training mockIDs only
for i in range(25):
    mockIDs.append('%.2d' %i)
    if '%.2d'%i not in Test_mockIDs: 
        Train_mockIDs.append('%.2d' %i)
mockIDs.append('fid')

CS_DIR = '/home/bengib/cosmoSLICS/'          # cosmoSLICS directory
Realisations = 25

# Read in the cosmological parameters                                                                                         
# These will be the output of the neural net
num_pCosmol = 4 # The number of cosmol. params to read in and predict with the CNN
Cosmols_Raw = np.loadtxt(CS_DIR+'/cosmoSLICS_Cosmologies_Omm-S8-h-w0-sig8-sig8bf.dat')[:,0:num_pCosmol] # Only take the frst n columns

Read_DIR = 'QuickReadData/SplitType-%s' %Split_Type
if Fast_Or_Slow_Read == "Slow":
    print("Performing a slow read of the input data!")
    # Slow read of data takes ~140s PER z-bin on cuillin head node
    Shear, Train_Shear, Test_Shear, Cosmols, Train_Cosmols, Test_Cosmols, Train_Cosmols_IDs, Test_Cosmols_IDs = Slow_Read(CS_DIR,
                                                                                                                          mock_Type,
                                                                                                                          Cosmols_Raw,
                                                                                                                          mockIDs,
                                                                                                                          Train_mockIDs,
                                                                                                                          Test_mockIDs,
                                                                                                                          Realisations,
                                                                                                                          ZBlabel,
                                                                                                                          Res,
                                                                                                                          Test_Realisation_num,
                                                                                                                          Noise)
                                                                                                                          
    # Pickle the train/test data to make reading faster next time
    if not os.path.exists(Read_DIR):
        os.makedirs(Read_DIR)
    # Save shear
    np.save('QuickReadData/Shear_numz%s_SN%s_Res%s' %(len(ZBlabel), Noise, Res), Shear )
    np.save('%s/Train_Shear_numz%s_SN%s_Res%s' %(Read_DIR, len(ZBlabel), Noise, Res), Train_Shear )
    np.save('%s/Test_Shear_numz%s_SN%s_Res%s' %(Read_DIR, len(ZBlabel), Noise, Res), Test_Shear )
    # Save cosmols
    np.save('QuickReadData/Cosmol_numz%s_numPCosmol%s' %(len(ZBlabel), num_pCosmol), Cosmols )
    np.save('%s/Train_Cosmol_numz%s_numPCosmol%s' %(Read_DIR, len(ZBlabel), num_pCosmol), Train_Cosmols )
    np.save('%s/Test_Cosmol_numz%s_numPCosmol%s' %(Read_DIR, len(ZBlabel), num_pCosmol), Test_Cosmols )
    # Save cosmol IDs
    np.savetxt('%s/Train_Cosmol_numz%s.txt' %(Read_DIR, len(ZBlabel)),
               Train_Cosmols_IDs, header='Cosmol ID for map in Train set', fmt='%s')
    np.savetxt('%s/Test_Cosmol_numz%s.txt' %(Read_DIR, len(ZBlabel)),
               Test_Cosmols_IDs, header='Cosmol ID for map in Test set', fmt='%s')
    
    
elif Fast_Or_Slow_Read == "Fast":
    # Takes only ~47s for 5 z-bins
    print("Performing a quick read of the input data.")
    t1 = time.time()
    # Read pickled shear
    Shear = np.load('QuickReadData/Shear_numz%s_SN%s_Res%s.npy' %(len(ZBlabel), Noise, Res))
    Train_Shear = np.load('%s/Train_Shear_numz%s_SN%s_Res%s.npy' %(Read_DIR, len(ZBlabel), Noise, Res))
    Test_Shear = np.load('%s/Test_Shear_numz%s_SN%s_Res%s.npy' %(Read_DIR, len(ZBlabel), Noise, Res))
    # Read pickled Cosmols
    Cosmols = np.load('QuickReadData/Cosmol_numz%s_numPCosmol%s.npy' %(len(ZBlabel), num_pCosmol))
    Train_Cosmols = np.load('%s/Train_Cosmol_numz%s_numPCosmol%s.npy' %(Read_DIR, len(ZBlabel), num_pCosmol))
    Test_Cosmols = np.load('%s/Test_Cosmol_numz%s_numPCosmol%s.npy' %(Read_DIR, len(ZBlabel), num_pCosmol))
    # Read pickled Cosmol ID arrays
    Train_Cosmols_IDs = np.genfromtxt('%s/Train_Cosmol_numz%s.txt' %(Read_DIR, len(ZBlabel)), dtype='str' )
    Test_Cosmols_IDs =	np.genfromtxt('%s/Test_Cosmol_numz%s.txt' %(Read_DIR, len(ZBlabel)), dtype='str' )
    t2 = time.time()
    print("Quick read of data took %.0f s for %s redshift bins." %((t2-t1),len(ZBlabel)) )

                           
import torch          # main neural net module
import torch.nn as nn
import torch.nn.functional as F
#import torchvision    # This is a handy module used to load data of various forms
#import torchvision.transforms as transforms

# Apply Transforms to data
# Define the transform that will be applied to the input data
#transform = transforms.Compose(
#    [transforms.ToTensor(),
#     transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))])


# A mean and stan-dev for each channel - used to normalise data
Mean_T = np.zeros(Train_Shear.shape[1])
Std_T = np.zeros(Train_Shear.shape[1])
for i in range(Train_Shear.shape[1]):
    Mean_T[i] = np.mean( Train_Shear[:,i,:,:] )
    Std_T[i] = np.std( Train_Shear[:,i,:,:] )

# Convert inputs to torch tensors
Train_Cosmols = torch.from_numpy( Train_Cosmols ) 
Test_Cosmols = torch.from_numpy( Test_Cosmols )

Train_Shear = torch.from_numpy( Transform_Data(Train_Shear, Mean_T, Std_T) ).double()
Test_Shear = torch.from_numpy( Transform_Data(Test_Shear, Mean_T, Std_T) ).double()


# DEFINE THE NEURAL NET
class Net(nn.Module):

    # This bit of code defines the architecture of the CNN
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(Train_Shear.shape[1], 128, 5)  # Defines the first convolution:
                                                              # (in_channel, out_channel, filter_size)
                                         
        self.fc1 = nn.Linear(128*124*124, num_pCosmol)        # 124 is the num of pxls on each side of the output activation map

    # ...and when called, this function executes the CNN
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = x.view(-1, 128*124*124)  # Last 3 numbers are output dims of previous conv.
        x = self.fc1(x)
        return x
net = Net()
net = net.float()
#output = net(Train_Shear.float())
#print("Input shape is ", Train_Shear.shape)
#print("Output shape is ", output.shape)

#sys.exit()

# Import the tools used to optimise the neural net performance
import random
import torch.optim as optim
criterion = nn.MSELoss()

#optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
optimizer = optim.Adam(net.parameters(), lr=0.0001)
# Feed the shear maps to the CNN in batches of size:
batch_size = 4    
t1 = time.time()
for epoch in range(1):
    running_loss = 0.0  # value of the loss that gets updated as it trains
    
    # Feed the training set maps to the neural net in batches of
    # at a time. Therefore, create a randomised array of indicies
    # (0-->999) in sets of 5.
    rand_idx = np.arange(0,Train_Shear.shape[0])
    np.random.seed(epoch) # Seed the randomisation, so it's reproducible
    random.shuffle( rand_idx )
    rand_idx = np.reshape( rand_idx, ( int(len(rand_idx)/batch_size), batch_size) )

    for i in range( rand_idx.shape[0] ):
        inputs = Train_Shear[ rand_idx[i] ]
        labels = Train_Cosmols[ rand_idx[i] ]

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs.float())
        loss = criterion(outputs.float(), labels.float())
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i % 10 == 9:    # print every 10 batches
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss/10 ))
            #print( labels )
            print( outputs )
            running_loss = 0.0

t2 = time.time()
print('Finished Training. Took %.1f seconds.' %(t2-t1))


# NOW TEST THE PERFORMANCE OF THE CNN USING THE TEST SET

# Let's still feed maps to the CNN in batches
# so we make another array of input indices
# (doesn't need to be in random order as we're not training now).
rand_idx_test = np.arange(0,Test_Shear.shape[0])
rand_idx_test = np.reshape( rand_idx_test, ( int(len(rand_idx_test)/batch_size), batch_size) )
# This is an array to store the outputs of the CNN
Test_Cosmols_Pred = np.zeros([ Test_Shear.shape[0], Test_Cosmols.shape[1] ])
for i in range( rand_idx_test.shape[0] ):
    inputs = Test_Shear[ rand_idx_test[i] ]
    labels = Test_Cosmols[ rand_idx_test[i] ]
    outputs = net(inputs.float())
    # Store the output predictions
    Test_Cosmols_Pred[i*batch_size:(i+1)*batch_size, :] = outputs.detach().numpy()

# Plot the accuracy results    
Plot_Accuracy( Test_Cosmols_Pred, Test_Cosmols.numpy() )
