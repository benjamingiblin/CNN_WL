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
import random
from Functions_4_CNN import Slow_Read, Transform_Data, Untransform_Data, Avg_Pred, Plot_Accuracy, Plot_Accuracy_vs_nlayers

import torch          # main neural net module
import torch.nn as nn
import torch.nn.functional as F
# find GPU device if available, else use CPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

Train_CNN = True              # If True it will train the CNN (obviously)
                              # If False it will read a pre-saved CNN

mock_Type = "KV450"
Data_Type = "Shear"           # Ultimately may use kappa maps as well
                              # in which case this variable will change.
                              # !!! Note, still need to code in this functionality !!!
                            
ZBlabel = ['ZBcut0.1-0.3', 'ZBcut0.3-0.5', 'ZBcut0.5-0.7', 'ZBcut0.7-0.9', 'ZBcut0.9-1.2']
          #['ZBcut0.1-1.2']   # The redshift range imposed on the maps
          
Augment = True                # If True, will augment the training&test data sets by
                              # reading in rotated & reflected versions of the maps.
                              # !!! Note, still need to code in this functionality !!!
          
Noise = "None"                # The shape noise level in the maps
Res = 128                     # The number of pxls on each side of the maps
nclayers = int(sys.argv[1])                  # The number of conv. layers to have in the CNN
                              # The number of fully connected (FC) layers is currently fixed to 1.
print("Setting a CNN with %s conv layers." %nclayers)
                              
RUN = 0                       # Variable for checking convergence of CNN predictions
                              # run the CNN several times changing only this variable
                              # and compare output to verify convergence.

Fast_Or_Slow_Read = "Fast"    # if "Slow", reads all shear maps one-by-one into array
                              # (always run "Slow" first time).
                              # If "Fast", reads a pickled version of the data
                              
                           
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
mockIDs = []                  # store all mockIDS
Train_mockIDs = []            # store the training mockIDs only
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

# Establish the in/output subdirectory anme, where the trained CNN & data will be saved (or loaded from).
# Make the output directory different for each mock suite, data tpye (shear/kappa),
# split done on the data, if the data set was augmented, num. zbins, noiseless/noisy maps, num pxls on each map size (Res),
Data_keyname = 'Mock%s_Data%s_Split%s_zBins%s_Noise%s_Aug%s_Res%s' %(mock_Type,Data_Type,Split_Type,
                                                                     len(ZBlabel),Noise,
                                                                     Augment,Res)
Read_DIR = 'QuickReadData/%s' %Data_keyname
if Fast_Or_Slow_Read == "Slow":
    print("Performing a slow read of the input data!")
    # Slow read of data takes ~140s PER z-bin on cuillin head node
    Shear,Train_Shear,Test_Shear, Cosmols,Train_Cosmols,Test_Cosmols, Cosmols_IDs,Train_Cosmols_IDs,Test_Cosmols_IDs = Slow_Read(CS_DIR,       mock_Type,Cosmols_Raw,mockIDs,Train_mockIDs,Test_mockIDs,Realisations,ZBlabel,Res,Test_Realisation_num,Noise,Augment)
                                                                                                                          
    # Pickle the train/test data to make reading faster next time
    if not os.path.exists(Read_DIR):
        os.makedirs(Read_DIR)
    # Save shear
    np.save('%s/Shear' %Read_DIR, Shear )
    np.save('%s/Train_Shear' %Read_DIR, Train_Shear )
    np.save('%s/Test_Shear'  %Read_DIR, Test_Shear )
    # Save cosmols
    np.save('%s/Cosmol_numPCosmol%s' %(Read_DIR, num_pCosmol), Cosmols )
    np.save('%s/Train_Cosmol_numPCosmol%s' %(Read_DIR,num_pCosmol), Train_Cosmols )
    np.save('%s/Test_Cosmol_numPCosmol%s' %(Read_DIR, num_pCosmol), Test_Cosmols )
    # Save cosmol IDs
    np.savetxt('%s/Cosmol_IDs.txt' %Read_DIR,
               Cosmols_IDs, header='Cosmol ID for map in Shear set', fmt='%s')
    np.savetxt('%s/Train_Cosmol_IDs.txt' %Read_DIR,
               Train_Cosmols_IDs, header='Cosmol ID for map Train_Shear set', fmt='%s')
    np.savetxt('%s/Test_Cosmol_IDs.txt' %Read_DIR,
               Test_Cosmols_IDs, header='Cosmol ID for map in Test_Shear set', fmt='%s')
    
    
elif Fast_Or_Slow_Read == "Fast":
    # For shear only, takes 47s for 5 bins & Augment=False,
    # 337s for 5 zbins & Agument=True,
    print("Performing a quick read of the input data.")
    t1 = time.time()
    # Read pickled shear
    Shear = np.load('%s/Shear.npy' %Read_DIR)
    Train_Shear = np.load('%s/Train_Shear.npy' %Read_DIR)
    Test_Shear = np.load('%s/Test_Shear.npy'  %Read_DIR)
    # Read pickled Cosmols
    Cosmols = np.load('%s/Cosmol_numPCosmol%s.npy' %(Read_DIR, num_pCosmol))
    Train_Cosmols = np.load('%s/Train_Cosmol_numPCosmol%s.npy' %(Read_DIR,num_pCosmol))
    Test_Cosmols = np.load('%s/Test_Cosmol_numPCosmol%s.npy' %(Read_DIR, num_pCosmol))
    # Read Cosmol ID arrays
    Cosmols_IDs = np.genfromtxt('%s/Cosmol_IDs.txt' %Read_DIR, dtype='str')
    Train_Cosmols_IDs = np.genfromtxt('%s/Train_Cosmol_IDs.txt' %Read_DIR, dtype='str' )
    Test_Cosmols_IDs =	np.genfromtxt('%s/Test_Cosmol_IDs.txt' %Read_DIR, dtype='str' )
    t2 = time.time()
    print("Quick read of data took %.0f s for %s redshift bins." %((t2-t1),len(ZBlabel)) )


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


# Define the padding, stride & filter sizes to be used in the CNN layers
# and define a function to compute the activation map size from these.

def Calc_Output_Map_Size(initial_size, F, P, S):
    output_size = float(initial_size - F + 2*P) / S
    if output_size == int(output_size):
        # if integer, add one
        output_size += 1
    else:
        # if non-integer, need to add 0.5
        output_size += 0.5
    return int(output_size)

conv1_padding=0 # padding applied on 1st conv layer
conv1_filter =5 # filter size of first conv layer (also subsequent layers)
conv1_stride=1  # stride used in first conv layer (also subsequent layers)  

# calculate the size of the map outputted by the 1st conv layer
act1_map_size = Calc_Output_Map_Size( Res, conv1_filter, conv1_padding, conv1_stride )
# then apply a padding of 2, with unchanged filter size & stride, to subequent layers to keep this map size unchanged:
conv2_padding = 2
conv2_filter = conv1_filter
conv2_stride = conv1_stride
# define the (for now) constant output channel size of the convolutional layers.
output_channel = 128

# DEFINE THE NEURAL NET
class Net(nn.Module):

    # This bit of code defines the architecture of the CNN
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(Train_Shear.shape[1], output_channel, conv1_filter,
                               padding=conv1_padding, stride=conv1_stride)  # Defines the first convolution:
                                                                            # (in_channel, out_channel, filter_size)

        self.conv2 = nn.Conv2d(output_channel, output_channel, conv2_filter,
                               padding=conv2_padding, stride=conv2_stride)
        self.fc1 = nn.Linear(output_channel*act1_map_size*act1_map_size, num_pCosmol)
                               # act1_map_size is the num pxls on each side of the activation map output from conv layer 1
                               # and we have chosen the padding on conv2 to keep this unchanged,
                               # hence output_channel*act1_map_size^2 is the dimensions of the activ. map from conv2
        
    # ...and when called, this function executes the CNN
    def forward(self, x):
        x = F.relu(self.conv1(x))
        # Put the output map through conv2 as many times as nlayers dictates.
        # Note, as long as padding=2 and stride=1, size of output maps will stay constant (124*124)
        for nl in range(nclayers-1):
            x = F.relu(self.conv2(x))
        x = x.view(-1, x.shape[1]*x.shape[2]*x.shape[3])      # Last 3 numbers are output dims of previous conv.
        x = self.fc1(x)
        return x
net = Net()
net.to(device).float()
#output = net(Train_Shear[0:5,:].to(device).float())
#print("Input shape is ", Train_Shear.shape)
#print("Output shape is ", output.shape)

# Set the up the save directory for the results -
# depends on all parameters of the data (Data_keyname)
# and the number of layers in the CNN:
Out_DIR = 'Results_CNN/%s/Net_convlayers%s_FClayers1' %(Data_keyname,nclayers)
if not os.path.exists(Out_DIR):
    os.makedirs(Out_DIR)

# Make a keyword containing the parameters of the CNN architecture
# filter size (F), padding (P), stride (S) in the 1st (conv1) & subsequent conv layers (conv2)
# and save the number of output cosmological parameters predicted by the CNN.
Arch_keyname = 'conv1F%sP%sS%s-conv2F%sP%sS%s_FC%sCosmol_run%s' %(conv1_filter,conv1_padding,conv1_stride,
                                                                  conv2_filter,conv2_padding,conv2_stride,
                                                                  num_pCosmol, RUN)
    
# Whether training or just testing, feed the shear maps to the CNN in batches of size:
batch_size = 4


if Train_CNN:
    import torch.optim as optim
    criterion = nn.MSELoss()

    #optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
    optimizer = optim.Adam(net.parameters(), lr=0.0001)

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
            inputs = Train_Shear[ rand_idx[i] ].to(device)
            labels = Train_Cosmols[ rand_idx[i] ].to(device)

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
                #print( outputs )
                running_loss = 0.0

    t2 = time.time()
    print('Finished Training. Took %.1f seconds.' %(t2-t1))
    # Save the CNN
    torch.save(net.state_dict(), '%s/Net_%s.pth' %(Out_DIR,Arch_keyname))

else:
    # Don't train, load a pre-trained CNN
    net = Net()
    net.to(device).float()
    net.load_state_dict(torch.load('%s/Net_%s.pth' %(Out_DIR,Arch_keyname)))



# NOW TEST THE PERFORMANCE OF THE CNN USING THE TEST SET

# Let's still feed maps to the CNN in batches
# so we make another array of input indices
# (doesn't need to be in random order as we're not training now).
rand_idx_test = np.arange(0,Test_Shear.shape[0])
rand_idx_test = np.reshape( rand_idx_test, ( int(len(rand_idx_test)/batch_size), batch_size) )
# This is an array to store the outputs of the CNN
Test_Cosmols_Pred = np.zeros([ Test_Shear.shape[0], Test_Cosmols.shape[1] ])
for i in range( rand_idx_test.shape[0] ):
    inputs = Test_Shear[ rand_idx_test[i] ].to(device)
    labels = Test_Cosmols[ rand_idx_test[i] ].to(device)
    outputs = net(inputs.float())
    # Store the output predictions
    Test_Cosmols_Pred[i*batch_size:(i+1)*batch_size, :] = outputs.detach().cpu().numpy()

# The CNN has produced many predictions at each cosmology.
# Compute an avg prediction per cosmology and corresponding error.
# Provide it with the corresponding test cosmols, so it can sort the predictions by what cosmology they correspond to.
Test_Cosmols_Pred_Avg, Test_Cosmols_Pred_Err, Test_Cosmols_Unique = Avg_Pred(Test_Cosmols_Pred, Test_Cosmols.numpy() )
# Save output predictions:
np.save('%s/TestPredAvg_%s' %(Out_DIR,Arch_keyname), Test_Cosmols_Pred_Avg)
np.save('%s/TestPredErr_%s' %(Out_DIR,Arch_keyname), Test_Cosmols_Pred_Err)
np.save('%s/TestTrue'%Out_DIR, Test_Cosmols_Unique )

# Plot the accuracy results
savename = '%s/PlotAcc_%s.png' %(Out_DIR, Arch_keyname)
#Plot_Accuracy( Test_Cosmols_Pred_Avg, Test_Cosmols_Pred_Err,
#               Test_Cosmols_Unique,
#               Test_mockIDs, savename)


Run_Multilayers = False
if Run_Multilayers:
    # Read in the results for multiple CNN architectures,
    # i.e. changing in number of conv layers, and plot how the accuracy changes nclayers:
    multi_layers = np.arange(1,11)
    Test_Cosmols_Pred_Avg_Stack = np.zeros([ Test_Cosmols_Pred_Avg.shape[0],
                                         Test_Cosmols_Pred_Avg.shape[1], len(multi_layers) ])
    Test_Cosmols_Pred_Err_Stack = np.zeros_like( Test_Cosmols_Pred_Avg_Stack )
    for i in range( len(multi_layers) ):
        tmp_Out_DIR = Out_DIR.split('convlayers')[0] + 'convlayers%s_FC'%multi_layers[i] + Out_DIR.split('FC')[-1]
        Test_Cosmols_Pred_Avg_Stack[:,:,i] = np.load('%s/TestPredAvg_%s.npy' %(tmp_Out_DIR,Arch_keyname))
        Test_Cosmols_Pred_Err_Stack[:,:,i] = np.load('%s/TestPredErr_%s.npy' %(tmp_Out_DIR,Arch_keyname))

    multi_savename = '%s/Plot-Nclayers%s-%s_%s.png' %(tmp_Out_DIR, multi_layers[0], multi_layers[1], Arch_keyname)
    Plot_Accuracy_vs_nlayers(multi_layers,
                             Test_Cosmols_Pred_Avg_Stack,
                             Test_Cosmols_Pred_Err_Stack,
                             Test_Cosmols_Unique,
                             Test_mockIDs,
                             multi_savename)
