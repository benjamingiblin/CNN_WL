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
import sys                                        # Used to exit code w/o error & read in inputs from command line    
import os
import random
from scipy.ndimage import gaussian_filter as gauss
from Classes_4_CNN import Linear_Net, Res_Net, Train_CNN_Class, Test_CNN_Class
from Functions_4_CNN import Slow_Read, Transform_Data, Untransform_Data, Calc_Output_Map_Size #, Train_CNN, Test_CNN, Transform_And_Train
from Functions_4_CNN import Apply_Smoothing_To_Shear_Channels, Avg_Pred, Plot_Accuracy, Plot_Accuracy_vs_Q, Plot_Pred_vs_Q

import torch          # main neural net module
import torch.nn as nn
import torch.nn.functional as F
# find GPU device if available, else use CPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

Perform_Train = True               # If True it will train the CNN (obviously)
                                   # If False it will read a pre-saved CNN
Perform_Test = True                # If True, runs the CNN on the test cosmologies
                                   # every 5 epochs in training, and at the end of training.
                                   # If False it reads in the pre-saved results

Net_Type = "Linear" # If Linear, then conv layers are applied sequentially (no residual layers).
                      # If Residual, then maps pass through residual blocks (conv+actv+conv)

mock_Type = "KV450" #"KV450" #"KiDS1000"
if mock_Type == "KV450":
    Realisations = 25
elif mock_Type== "KiDS1000":
    Realisations = 10

Data_Type = "ShearKappa"           # "Shear", "Kappa" or "ShearKappa"
                                                          
ZBlabel = ['ZBcut0.1-0.3', 'ZBcut0.3-0.5', 'ZBcut0.5-0.7', 'ZBcut0.7-0.9', 'ZBcut0.9-1.2']
          #['ZBcut0.1-1.2']   # The redshift range imposed on the maps
neff =    [0.62, 1.18, 1.85, 1.26, 1.31]        # The num of gals per arcmin^2 in each redshift bin
sigma_e = [ 0.27, 0.258, 0.273, 0.254, 0.27 ]   # galaxy dispersion in each redshift bin
          # We need these two arrays to generate shape noise realisations if Noise_Cycle is True.
          
Augment_Train = True          # If True, within the training loop, it will perform rotations/reflections
                              # of each map, to help the CNN learn cosmol. is invariant to these.
Augment_Data = False          # Leave False. If True, it reads rot'd/ref'd maps into memory.
                              # This is redundant if Augment_Train is True, and former takes less memory.
          
Noise_Data = "None"           # The shape noise level in the maps you read in: "None"/"On"
Noise_Cycle = False            # If True, it will make multiple shape noise realistions
                              # of each shear-only map, and cycle through these in training.
N_Noise = 20                  # The number of shape noise realisations to do if Noise_Cycle is True.
if Noise_Cycle==True:
    # Create a Noise_SaveTag string to designate how many noise realisations were used in Training
    Noise_SaveTag = "Cycle%s" %N_Noise
    
    if Noise_Data=="On":
        print("You cannot have Noise_Data 'On' and Noise_Cycle 'True'. One or the other but not both. EXITING.")
        sys.exit()
else:
    N_Noise = 1
    Noise_SaveTag = Noise_Data # either "On" (fixed finite shape noise) or "None"

# If we include shape noise, need smaller learning rate(?)
if Noise_SaveTag == "None":
    lr=5e-6
else:
    lr=5e-6          # tried doen to 1e-8, get problems with fixed shape noise.        
    
Res = 128                     # The number of pxls on each side of the maps
nclayers = int(sys.argv[1])   # The number of conv. layers to have in the CNN
                              # The number of fully connected (FC) layers is currently fixed to 1.
conv1_filter = int(sys.argv[2]) 
                              
print("Setting up a CNN with %s conv layers." %nclayers)

Epochs_Start = int(sys.argv[3]) # If zero, starts a CNN from fresh.
                                # If>0, reads in a pre-trained CNN & starts training that.
Epochs_End = int(sys.argv[4])   # where to stop training.
Epochs_Tot = Epochs_End - Epochs_Start

Fast_Or_Slow_Read = "Slow"    # if "Slow", reads all shear maps one-by-one into array
                              # (always run "Slow" first time).
                              # If "Fast", reads a pickled version of the data
                              
                           
Split_Type = "TestAll"            # "Fid" meaning fiducial, means the 6 cosmols closest to the middle
                              # of the parameter space are the test cosmologies, with the first 12
                              # LOS per seed being test maps, with the final 13 per seed being training maps.
                           
if Split_Type == "Fid":
    Test_mockIDs = ['00','05','07','12','18','fid']     # the test mock IDs
    Test_Realisation_num = int(Realisations/2)  # Use this many of the lines of sight in Test set
                              # So the other (Realisations-Test_Realisation_num) will be in the training set
                              
elif Split_Type == "TestAll":
    # Put a few realisations from ALL cosmologies in the Test set.
    Test_mockIDs = []
    for i in range(25):
        Test_mockIDs.append('%.2d' %i)
    Test_mockIDs.append('fid')
    Test_Realisation_num = 5
    
elif Split_Type == "None":
    # Then there is no splitting of the train & test sets, everything is put in the training set.
    Test_mockIDs = []
    Test_Realisation_num = 0

else:
    print("The only Split_Type's currently supported are: Fid, TestAll, None. Please set accordingly. EXITING.")
    sys.exit()

                              
# Assemble a list of the cosmoSLICS 26 IDs: ['fid', '00', '01',...,'24']
mockIDs = []                  # store all mockIDS
Train_mockIDs = []            # store the training mockIDs only
for i in range(25):
    mockIDs.append('%.2d' %i)
    if '%.2d'%i not in Test_mockIDs: 
        Train_mockIDs.append('%.2d' %i)
mockIDs.append('fid')
if 'fid' not in Test_mockIDs:
    Train_mockIDs.append('fid')

CS_DIR = '/home/bengib/cosmoSLICS/'          # cosmoSLICS directory


# Read in the cosmological parameters                                                                                         
# These will be the output of the neural net
num_pCosmol = 4 # The number of cosmol. params to read in and predict with the CNN
Cosmols_Raw = np.loadtxt(CS_DIR+'/cosmoSLICS_Cosmologies_Omm-S8-h-w0-sig8-sig8bf.dat')[:,0:num_pCosmol]
                

# Establish the in/output subdirectory anme, where the trained CNN & data will be saved (or loaded from).
# Make the output directory different for each mock suite, data tpye (shear/kappa),
# split done on the data, if the data set was augmented, num. zbins, noiseless/noisy maps, num pxls on each map size (Res),
Data_keyname = 'Mock%s_Data%s_Split%s_zBins%s_Noise%s_Aug%s_Res%s' %(mock_Type,Data_Type,Split_Type,
                                                                     len(ZBlabel),Noise_Data,
                                                                     Augment_Data,Res)
Read_DIR = 'QuickReadData/%s' %Data_keyname
if Fast_Or_Slow_Read == "Slow":
    print("Performing a slow read of the input data!")
    # Slow read of data takes ~140s PER z-bin on cuillin head node
    Data,Train_Data,Test_Data, Cosmols,Train_Cosmols,Test_Cosmols, Cosmols_IDs,Train_Cosmols_IDs,Test_Cosmols_IDs = Slow_Read(CS_DIR, Data_Type,mock_Type,Cosmols_Raw,mockIDs,Train_mockIDs,Test_mockIDs,Realisations,ZBlabel,Res,Test_Realisation_num,Noise_Data,False)
                                                                                                                          
    # Pickle the train/test data to make reading faster next time
    if not os.path.exists(Read_DIR):
        os.makedirs(Read_DIR)
    # Save shear
    np.save('%s/Data' %Read_DIR, Data )
    np.save('%s/Train_Data' %Read_DIR, Train_Data )
    np.save('%s/Test_Data'  %Read_DIR, Test_Data )
    # Save cosmols
    np.save('%s/Cosmol_numPCosmol%s' %(Read_DIR, num_pCosmol), Cosmols )
    np.save('%s/Train_Cosmol_numPCosmol%s' %(Read_DIR,num_pCosmol), Train_Cosmols )
    np.save('%s/Test_Cosmol_numPCosmol%s' %(Read_DIR, num_pCosmol), Test_Cosmols )
    # Save cosmol IDs
    np.savetxt('%s/Cosmol_IDs.txt' %Read_DIR,
               Cosmols_IDs, header='Cosmol ID for map in Overall data set', fmt='%s')
    np.savetxt('%s/Train_Cosmol_IDs.txt' %Read_DIR,
               Train_Cosmols_IDs, header='Cosmol ID for map Train_Data set', fmt='%s')
    np.savetxt('%s/Test_Cosmol_IDs.txt' %Read_DIR,
               Test_Cosmols_IDs, header='Cosmol ID for map in Test_Data set', fmt='%s')
    
elif Fast_Or_Slow_Read == "Fast":
    # For shear only, takes 47s for 5 bins & Augment_Data=False,
    # 337s for 5 zbins & Agument_Data=True,
    print("Performing a quick read of the input data.")
    t1 = time.time()
    # Read pickled shear
    Data = np.load('%s/Data.npy' %Read_DIR)
    Train_Data = np.load('%s/Train_Data.npy' %Read_DIR)
    Test_Data = np.load('%s/Test_Data.npy'  %Read_DIR)
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

    
if Fast_Or_Slow_Read != "None":
    # Then it means the data was read in, proceed to transform it:
    # Apply Gaussian smoothing of maps
    sigma_pxl=1 # smoothing scale in units of pxl
    Data = Apply_Smoothing_To_Shear_Channels(Data_Type, Data, sigma_pxl)
    Train_Data = Apply_Smoothing_To_Shear_Channels(Data_Type, Train_Data, sigma_pxl)
    Test_Data = Apply_Smoothing_To_Shear_Channels(Data_Type, Test_Data, sigma_pxl)
    
    input_channel = Train_Data.shape[1]
    # A mean and stan-dev for each channel - used to normalise data
    Mean_T = np.zeros(input_channel)
    Std_T = np.zeros(input_channel)
    for i in range(input_channel):
        Mean_T[i] = np.mean( Train_Data[:,i,:,:] )
        Std_T[i] = np.std( Train_Data[:,i,:,:] )

    # Convert inputs to torch tensors
    Train_Cosmols = torch.from_numpy( Train_Cosmols ) 
    Test_Cosmols = torch.from_numpy( Test_Cosmols )

    Train_Data = torch.from_numpy( Transform_Data(Train_Data, Mean_T, Std_T) ).double()
    Test_Data = torch.from_numpy( Transform_Data(Test_Data, Mean_T, Std_T) ).double()
    
    # Save the mean & stdev so the KiDS-1000 data can be transformed in the same way:
    np.savetxt('%s/Channel_Mean_Stdev.txt' %Read_DIR, np.c_[Mean_T, Std_T], header='# channel mean, sigma')
    
else:
    # need to define the input channel to define the neural network
    # and read in a pre-saved one. This depends on the Data_Type.    
    if Data_Type == "Kappa":
        input_channel = 1*len(ZBlabel)
    elif Data_Type == "Shear":
        input_channel =	2*len(ZBlabel)
    elif Data_Type == "ShearKappa":
        input_channel = 3*len(ZBlabel)

# Define the padding, stride & filter sizes to be used in the CNN layers
conv1_padding=0                # padding applied on 1st conv layer
conv1_stride=1                 # stride used in first conv layer (also subsequent layers)  
pool_layers = np.array([2,4])  #np.array([2,4])
                               # applying pooling after these convolutions
                               # not using zero-based indexing here (2=the 2nd conv applied).

# calculate the size of the map outputted by the 1st conv layer
# also returns the required padding to keep the actv. maps constant size.
# Note: this will only maintain map size if filter_size is odd.
# Otherwise the maps will get smaller by 1 pixel after every convolution, breaking the Net function below.
act1_map_size, conv2_padding = Calc_Output_Map_Size( Res,
                                                     conv1_filter,conv1_padding,conv1_stride,
                                                     pool_layers, nclayers)
conv2_filter = conv1_filter
conv2_stride = conv1_stride
# define the (for now) constant output channel size of the convolutional layers.
output_channel = 128

# Set up the CNN architecture:
if Net_Type == "Linear":
    net = Linear_Net(input_channel, output_channel,
                     conv1_filter,conv1_padding,conv1_stride,
                     conv2_filter,conv2_padding,conv2_stride,
                     act1_map_size,num_pCosmol,
                     nclayers, pool_layers)

elif Net_Type == "Residual":
    net = Res_Net(input_channel, output_channel,
                  conv1_filter,conv1_padding,conv1_stride,
                  conv2_filter,conv2_padding,conv2_stride,
                  act1_map_size,num_pCosmol,
                  nclayers, pool_layers)

net.to(device).float()

# Lines to test dimensionality of output from the CNN:
#output = net(Train_Data[0:5,:].to(device).float())
#print("Input shape is ", Train_Data.shape)
#print("Output shape is ", output.shape) 


# Set the up the save directory for the results -
# depends on all parameters of the data (Data_keyname),
# but modulated by whether the augmentation was performed in the training (Augment_Data-->Augment_Train),
# whether there was multiple shape noise realisations in the training (Noise_Data-->Noise_SaveTag),
# and on the number of layers in the CNN:
Train_keyname = 'Mock%s_Data%s_Split%s_zBins%s_Noise%s_Aug%s_Res%s' %(mock_Type,Data_Type,Split_Type,
                                                                     len(ZBlabel),Noise_SaveTag,
                                                                     Augment_Train,Res)
Out_DIR = 'Results_CNN/%s/%sNet_convlayers%s_FClayers1' %(Train_keyname,Net_Type,nclayers)
# add on to Out_DIR if pooling was performed, and if so, on what layers (PL):
PL="_PL"
if len(pool_layers)>0 and nclayers>=pool_layers.min():
    for i in range(1,nclayers+1):
        if i in pool_layers:
            PL+="%s-" %i
    PL=PL[:-1] # rm final hyphen
else:
    PL+="None"
Out_DIR+=PL

if not os.path.exists(Out_DIR):
    os.makedirs(Out_DIR)

# Make a keyword containing the parameters of the CNN architecture
# filter size (F), padding (P), stride (S) in the 1st (conv1) & subsequent conv layers (conv2)
# and save the number of output cosmological parameters predicted by the CNN.
Arch_keyname = 'conv1F%sP%sS%s-conv2F%sP%sS%s_FC%sCosmol_Epochs' %(conv1_filter,conv1_padding,conv1_stride,
                                                                  conv2_filter,conv2_padding,conv2_stride,
                                                                  num_pCosmol)
if Epochs_Start>0:
    # Read in a pre-trained CNN to either continue training or make predictions with.
    print('Pre-reading CNN: %s/Net_%s%s.pth' %(Out_DIR,Arch_keyname,Epochs_Start))
    net.load_state_dict( torch.load('%s/Net_%s%s.pth' %(Out_DIR,Arch_keyname,Epochs_Start), map_location=device) )
    net.to(device).float()


# Whether training or just testing, feed the shear maps to the CNN in batches of size:
batch_size = 4

if Perform_Train:
    import torch.optim as optim
    criterion = nn.MSELoss()
    optimizer = optim.Adam(net.parameters(), lr=lr)
    TrCNN = Train_CNN_Class(net, criterion, optimizer) 
    TeCNN = Test_CNN_Class(net, Test_Data, Test_Cosmols, batch_size)
    
    t1 = time.time()
    for epoch in range(Epochs_Start, Epochs_End):
        running_loss = 0.0  # value of the loss that gets updated as it trains
    
        # Feed the training set maps to the neural net in batches of
        # at a time. Therefore, create a randomised array of indicies
        # (0-->999) in sets of 5.
        rand_idx = np.arange(0,Train_Data.shape[0])
        np.random.seed(1) # Seed the randomisation, so it's reproducible
        random.shuffle( rand_idx )
        rand_idx = np.reshape( rand_idx, ( int(len(rand_idx)/batch_size), batch_size) )

        for i in range( rand_idx.shape[0] ):
            inputs = Train_Data[ rand_idx[i] ].to(device)
            labels = Train_Cosmols[ rand_idx[i] ].to(device)
            
            # Loop through shape noise realisations (does only one noiseless loop if Noise_Cycle is False):
            for n_noise in range(N_Noise):
                # train:
                #net, loss, optimizer = Train_CNN(net, criterion, optimizer, inputs, labels)
                loss = TrCNN.Train_CNN(inputs, labels)
                
                if Augment_Train:
                    #net, loss, optimizer = Transform_And_Train(net, criterion, optimizer, inputs, labels)
                    loss = TrCNN.Transform_And_Train(inputs, labels)
                    
            running_loss += loss.item()
            if i % 10 == 0:    # print every 10 batches
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss/10 ))
                #print( labels )
                #print( outputs )
                running_loss = 0.0


        if (epoch+1) % 20 == 0 and Perform_Test:
            # Make a prediction for the test cosmology every 5 epochs
            #tmp_Test_Pred = Test_CNN(net, Test_Data, Test_Cosmols, batch_size)
            tmp_Test_Pred = TeCNN.Test_CNN()
            tmp_Test_Avg, tmp_Test_Err, _ = Avg_Pred(tmp_Test_Pred, Test_Cosmols.numpy() )
            np.save('%s/TestPredAvg_%s%s' %(Out_DIR,Arch_keyname,epoch+1), tmp_Test_Avg) 
                
    t2 = time.time()
    print('Finished Training. Took %.1f seconds.' %(t2-t1))
    # Save the CNN
    torch.save(net.state_dict(), '%s/Net_%s%s.pth' %(Out_DIR,Arch_keyname,Epochs_End))

else:
    # Don't train, load a pre-trained CNN
    net.load_state_dict( torch.load('%s/Net_%s%s.pth' %(Out_DIR,Arch_keyname,Epochs_End), map_location=device) )


if Perform_Test:
    # NOW TEST THE PERFORMANCE OF THE CNN USING THE TEST SET
    #Test_Cosmols_Pred = Test_CNN(net, Test_Data, Test_Cosmols, batch_size)
    TeCNN = Test_CNN_Class(net, Test_Data, Test_Cosmols, batch_size)
    Test_Cosmols_Pred = TeCNN.Test_CNN()
    # Compute an avg prediction per cosmology and corresponding error:
    Test_Cosmols_Pred_Avg, Test_Cosmols_Pred_Err, Test_Cosmols_Unique = Avg_Pred(Test_Cosmols_Pred, Test_Cosmols.numpy() )
    # Save output predictions:
    np.save('%s/TestPredAvg_%s%s' %(Out_DIR,Arch_keyname,Epochs_End), Test_Cosmols_Pred_Avg)
    np.save('%s/TestPredErr_%s%s' %(Out_DIR,Arch_keyname,Epochs_End), Test_Cosmols_Pred_Err)
    np.save('%s/TestTrue'%Out_DIR, Test_Cosmols_Unique )

else:
    # Load the pre-saved test results
    Test_Cosmols_Pred_Avg = np.load('%s/TestPredAvg_%s%s.npy' %(Out_DIR,Arch_keyname,Epochs_End))
    Test_Cosmols_Pred_Err = np.load('%s/TestPredErr_%s%s.npy' %(Out_DIR,Arch_keyname,Epochs_End))
    Test_Cosmols_Unique   = np.load('%s/TestTrue.npy'%Out_DIR)



    
# ---------------------------------------- PLOTTING FUNCTIONS ----------------------------------------


# Plot the accuracy results  
savename = '%s/PlotAcc_%s%s.png' %(Out_DIR, Arch_keyname,Epochs_Tot)
#Plot_Accuracy( Test_Cosmols_Pred_Avg, Test_Cosmols_Pred_Err,
#               Test_Cosmols_Unique,
#               Test_mockIDs, savename) 


Run_Multilayers = False
if Run_Multilayers:
    # Read in the results for multiple CNN architectures,
    # i.e. changing in number of conv layers, and plot how the accuracy changes nclayers:
    #multi_layers = [1,2,3,4,5,6,7,8,9,10,15,20] #np.arange(1,11)
    multi_layers = [1,5,10,15,20]
    
    Test_Cosmols_Pred_Avg_Stack = np.zeros([ Test_Cosmols_Pred_Avg.shape[0],
                                         Test_Cosmols_Pred_Avg.shape[1], len(multi_layers) ])
    Test_Cosmols_Pred_Err_Stack = np.zeros_like( Test_Cosmols_Pred_Avg_Stack )

    for i in range( len(multi_layers) ):
        tmp_Out_DIR = Out_DIR.split('convlayers')[0] + 'convlayers%s_FC'%multi_layers[i] + Out_DIR.split('FC')[-1]
        Test_Cosmols_Pred_Avg_Stack[:,:,i] = np.load('%s/TestPredAvg_%s.npy' %(tmp_Out_DIR,Arch_keyname))
        Test_Cosmols_Pred_Err_Stack[:,:,i] = np.load('%s/TestPredErr_%s.npy' %(tmp_Out_DIR,Arch_keyname))

        
    multi_savename = '%s/Plot-Nclayers%s-%s_%s.png' %(tmp_Out_DIR, multi_layers[0], multi_layers[-1], Arch_keyname)
    ylimit_acc = [[-40.,40.], [-40.,40.], [-40.,40.], [-40.,40.]]
    ylimit_raw = [[0.2,0.39], [0.7,0.85], [0.69,0.72], [-1.4, -1.1]]
    Plot_Accuracy_vs_Q(multi_layers,
                       Test_Cosmols_Pred_Avg_Stack,
                       Test_Cosmols_Pred_Err_Stack,
                       Test_Cosmols_Unique,
                       Test_mockIDs,
                       r'$N_{\rm conv-layers}$',
                       multi_savename,
                       ylimit_acc)

    Plot_Pred_vs_Q(multi_layers,
                   Test_Cosmols_Pred_Avg_Stack,
                   Test_Cosmols_Pred_Err_Stack,
                   Test_Cosmols_Unique,
                   Test_mockIDs,
                   r'$N_{\rm conv-layers}$',
                   multi_savename,
                   ylimit_raw)


Run_Multifilters = False
if Run_Multifilters:
    # Plot the average accuracy for multiple filter sizes used in the convolutions
    multi_filters = [1,3,5,7,9]
    multi_padding = [0,1,2,3,4] # conv2 padding changes with filter-size to preserve actv. map size.
    Test_Cosmols_Pred_Avg_Stack = np.zeros([ Test_Cosmols_Pred_Avg.shape[0],
                                             Test_Cosmols_Pred_Avg.shape[1], len(multi_filters) ])
    Test_Cosmols_Pred_Err_Stack = np.zeros_like( Test_Cosmols_Pred_Avg_Stack )
    for i in range( len(multi_filters) ):
        tmp_Arch_keyname = 'conv1F%sP%sS%s-conv2F%sP%sS%s_FC%sCosmol_Epochs%s' %(multi_filters[i],conv1_padding,conv1_stride,
                                                                              multi_filters[i],multi_padding[i],conv2_stride,
                                                                              num_pCosmol, Epochs_Tot )
        Test_Cosmols_Pred_Avg_Stack[:,:,i] = np.load('%s/TestPredAvg_%s.npy' %(Out_DIR,tmp_Arch_keyname))
        Test_Cosmols_Pred_Err_Stack[:,:,i] = np.load('%s/TestPredErr_%s.npy' %(Out_DIR,tmp_Arch_keyname))

    multi_savename = '%s/Plot-Filter%s-%s_%s.png' %(Out_DIR, multi_filters[0], multi_filters[-1], Arch_keyname)
    ylimit_acc = [[-40.,40.], [-40.,40.], [-40.,40.], [-40.,40.]]
    Plot_Accuracy_vs_Q(multi_filters,
                       Test_Cosmols_Pred_Avg_Stack,
                       Test_Cosmols_Pred_Err_Stack,
                       Test_Cosmols_Unique,
                       Test_mockIDs,
                       r'Filter size [pxl]',
                       multi_savename, ylimit_acc)


Run_MultiEpochs = False
if Run_MultiEpochs:
    multi_epochs = np.arange(5,1005,5)
    # accidentally overwrote the 500 epoch training sample for ShearKappa 6 layer linear, remove from list:
    #multi_epochs = np.delete(multi_epochs, np.where(multi_epochs==500)[0])
    
    Test_Cosmols_Pred_Avg_Stack = np.zeros([ Test_Cosmols_Pred_Avg.shape[0],
                                             Test_Cosmols_Pred_Avg.shape[1], len(multi_epochs) ])
    Test_Cosmols_Pred_Err_Stack = np.zeros_like( Test_Cosmols_Pred_Avg_Stack )
    for i in range( len(multi_epochs) ):
        # read i data for various epochs
        tmp_Arch_keyname = 'conv1F%sP%sS%s-conv2F%sP%sS%s_FC%sCosmol_Epochs%s' %(conv1_filter,conv1_padding,conv1_stride,
                                                                              conv2_filter,conv2_padding,conv2_stride,
                                                                              num_pCosmol, multi_epochs[i])
        Test_Cosmols_Pred_Avg_Stack[:,:,i] = np.load('%s/TestPredAvg_%s.npy' %(Out_DIR,tmp_Arch_keyname))
        #Test_Cosmols_Pred_Err_Stack[:,:,i] = np.load('%s/TestPredErr_%s.npy' %(Out_DIR,tmp_Arch_keyname))

    # Renormalise the accuracies to be % differences from those obtained for the max num of epochs
    # need factor of Test_Cosmols_Unique into cancel factor in the Plot_Acc function.
    Test_Cosmols_Pred_Avg_Stack_Norm = np.copy(Test_Cosmols_Pred_Avg_Stack)
    Test_Cosmols_Pred_Err_Stack_Norm = np.copy(Test_Cosmols_Pred_Err_Stack)
    for i in range(Test_Cosmols_Pred_Avg_Stack.shape[2]):
        Test_Cosmols_Pred_Avg_Stack_Norm[:,:,i] = Test_Cosmols_Unique* Test_Cosmols_Pred_Avg_Stack[:,:,i] / Test_Cosmols_Pred_Avg_Stack[:,:,-1]
        Test_Cosmols_Pred_Err_Stack_Norm[:,:,i] = Test_Cosmols_Unique* Test_Cosmols_Pred_Err_Stack[:,:,i] / Test_Cosmols_Pred_Avg_Stack[:,:,-1]

    multi_savename = '%s/Plot-Epochs%s-%s_%s.png' %(Out_DIR, multi_epochs[0], multi_epochs[-1], Arch_keyname)
    ylimit_acc = [[-5.,15.], [-5.,5.], [-10.,9.9], [-25.,19.9]]
    Plot_Accuracy_vs_Q(multi_epochs,
                       Test_Cosmols_Pred_Avg_Stack,
                       Test_Cosmols_Pred_Err_Stack,
                       Test_Cosmols_Unique,
                       Test_mockIDs,
                       r'Number of epochs',
                       multi_savename, ylimit_acc)

    ylimit_raw = [[0.2,0.5], [0.6,0.85], [0.64,0.79], [-1.9, -0.95]]
    Plot_Pred_vs_Q(multi_epochs,
                   Test_Cosmols_Pred_Avg_Stack,
                   Test_Cosmols_Pred_Err_Stack,
                   Test_Cosmols_Unique,
                   Test_mockIDs,
                   r'Number of epochs',
                   multi_savename,
                   ylimit_raw)
