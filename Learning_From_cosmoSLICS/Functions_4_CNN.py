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
              Realisations, ZBlabel, Res, Test_Realisation_num, Noise, Augment):
    if Augment:
        # Reading in 3 rotations & 2 reflections of every map (6x larger training sample)
        aug_f=6
        aug_cycle = ['', '_Rot1','_Rot2','_Rot3', '_Ref1','_Ref2'] 
    else:
        # Reading in only the map itself
        aug_f=1
        aug_cycle = ['']

    if Noise == "None":
        SN = ["None", "None", "None", "None", "None"]
    elif Noise == "On":
        if len(ZBlabel)>1:
            # set it to KiDS1000 levels:
            SN = [0.27, 0.258, 0.273, 0.254, 0.27]
        else:
            # set it to the default value I used for KiDS-450:
            SN = [0.28]

    # Start preparing the arrays to store the shear maps & corresponding cosmological parameters:
    Shear = np.zeros([ len(mockIDs)*2*Realisations*4*aug_f, 2*len(ZBlabel), Res,Res ])
                                                              # This has the dimensions of 
                                                              # [num_maps, num_channels, pxl, pxl ]
                                                              # Here num_maps=26(cosmols)*2(seeds)*50(realisations)
                                                              # *4quadrants (* 6 transformations of each map)
                                                              # And num_channels (previously 3 for colours) is
                                                              # 2 for shear components * number of z-bins.

    Test_Shear = np.zeros([ len(Test_mockIDs)*2*Test_Realisation_num*4*aug_f, 2*len(ZBlabel), Res,Res ])
    Train_Shear = np.zeros([ ( len(Test_mockIDs)*(Realisations-Test_Realisation_num) + len(Train_mockIDs)*Realisations)*2*4*aug_f,
                         2*len(ZBlabel), Res,Res ])

    # Create arrays that will match the cosmol params to the shear images
    Cosmols = np.zeros([ Shear.shape[0], 4 ])              # store all cosmols
    Test_Cosmols = np.zeros([ Test_Shear.shape[0], 4 ])    # store the test cosmols
    Train_Cosmols = np.zeros([ Train_Shear.shape[0], 4 ])  # & the training cosmols
    # These arrays aren't used in the code. They're just for easily checking
    # the mockID that each row of the Shear arrays correspond to:
    Cosmols_IDs = []
    Test_Cosmols_IDs = []
    Train_Cosmols_IDs = []

    # ...Now that the arrays are all prepared, begin cycling through the data,
    # reading them in and sorting into either either Train or Test maps/cosmologies:

    t1 = time.time()                                    # Start the timer
    map_count=0                                         # counter for scrolling through map realisations (cosmol, seeds, los & quads)
    map_count_test=0
    map_count_train=0
    for i in range(len(mockIDs)):                    # Scroll through the mock IDs
        t1b = time.time()
        print( "On cosmology %s after %.1f s" %(i,(t1b-t1)) )
        for r in range(Realisations):                # Scroll through the 25 realisations
            for seed in ['a', 'f']:                  # Scroll through the 2 seeds 
                for q in range(4):                   # Scroll through the 4 quadrants
                    for ac in aug_cycle:             # Scroll through 6 transforms of each map (if Augment==True)
            
                        channel_count = 0                           # count for cycling through channels (shear components & zbins)
                        channel_count_test = 0
                        channel_count_train = 0
            
                        for zb in range(len(ZBlabel)):              # scroll through z-bins
                            for s in range(2):                      # Scroll through 2 shear components

                                # Open the map
                                f = fits.open(CS_DIR+'%s_%s/%s/Shear_Maps/Res%sx%s/%s/Shear%s_LOS%s_SN%s_Quad%s%s.fits'
                                    %(mockIDs[i],seed,mock_Type,Res,Res,ZBlabel[zb],s+1,r+1,SN[zb],q,ac))
                        
                                Shear[map_count,channel_count,:,:] = f[0].data # store shear
                                Cosmols[map_count,:] = Cosmols_Raw[i,:]        # and corresponding cosmols
                                Cosmols_IDs.append( mockIDs[i] )               # and the ID tag
                                
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

    return Shear, Train_Shear, Test_Shear, Cosmols, Train_Cosmols, Test_Cosmols, Cosmols_IDs, Train_Cosmols_IDs, Test_Cosmols_IDs


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


def Calc_Output_Map_Size(initial_size, F, P, S, pool_layers, nclayers):
    # calculate the size of the map outputted by the 1st conv layer,
    # & the padding needed to keep the actv. maps constant size with each subsequent conv.
    # If padding is specified by pool_layers,
    # then take into account how the actv map size reduces each time pooling is applied.
    # Note: this will only maintain map size if filter_size is odd.
    # Otherwise the maps will get smaller by 1 pixel after every convolution,
    # and this will break the Net() class defined in Classes_2_CNN.py (true as of 25/11/20).
    
    output_size = float(initial_size - F + 2*P) / S
    if output_size == int(output_size):
        # if integer, add one
        add = 1
    else:
        # if non-integer, need to add 0.5
        add = 0.5
    output_size += add

    # How many times are we pooling? This will reduce output dimensions by 1/2 per pool.
    # only count the pooling layers that are <= the number on conv layers:
    len_pool = len(np.where(pool_layers<=nclayers)[0])
    output_size = output_size * 0.5**len_pool

    # calculate the padding needed in subsequent conv's to maintain the map size.
    # this is a simple rearrangement of the above formula, setting in/out-put sizes equal.
    # NOTE: filter size must be odd for this to work.
    conv2_padding = int(0.5*((output_size-add)*S +F -output_size))
    return int(output_size), conv2_padding

def Train_CNN(net, criterion, optimizer, inputs, labels):
    # zero the parameter gradients
    optimizer.zero_grad()

    # forward + backward + optimize
    outputs = net(inputs.float())
    loss = criterion(outputs.float(), labels.float())
    loss.backward()
    optimizer.step()
    return loss, optimizer

def Test_CNN(net, Test_Data, Test_Labels, batch_size):
    # make minibatches of test data                                                                                                             
    rand_idx_test = np.arange(0,Test_Data.shape[0])
    rand_idx_test = np.reshape( rand_idx_test, ( int(len(rand_idx_test)/batch_size), batch_size) )
    # This is an array to store the outputs of the CNN
    Test_Pred = np.zeros([ Test_Data.shape[0], Test_Labels.shape[1] ])
    for i in range( rand_idx_test.shape[0] ):
        inputs = Test_Data[ rand_idx_test[i] ].to(device)
        labels = Test_Labels[ rand_idx_test[i] ].to(device)
        outputs = net(inputs.float())
        # Store the output predictions
        Test_Pred[i*batch_size:(i+1)*batch_size, :] = outputs.detach().cpu().numpy()
    return Test_Pred




def Generate_Shape_Noise(seed, batch_size, input_channel, Res, neff, sigma_e):
    f = int( input_channel / len(neff) ) # (1,2,3) for (Kappa-only, Shear-only, Kappa+Shear)
    neff = np.repeat(neff, f)            # neff per input channel
    sigma_e = np.repeat(sigma_e,f)       # sigma_e per input channel
    pxl_size = (5.*60. / Res)**2         # in arcmin^2 (5deg on side, assumes map split into quads).
    sigma_noise = sigma_e / np.sqrt( pxl_size*neff )

    # make shape noise realisations:
    noise = np.zeros([batch_size, input_channel, Res, Res])
    for i in range( input_channel ):
        np.random.seed(seed+i)       # different noise per channel
        noise[:,i,:,:] = np.random.normal(0, sigma_noise[i], [batch_size,Res,Res])
    return noise





def Avg_Pred(pred,test):
    
    # This read in the test set predictions for every single realisation and cosmology,
    # and converts these to an avg & mean-err per cosmology.
    # reads in the (true) test cosmologies so it can figure out which predictions it needs to avg together.

    # Find the unique values of Omega_m, to figure out how many test cosmologies there are
    indices = np.unique( test[:,0], return_index=True )[1]
    test_unique = test[ np.sort(indices), : ] # the unique test cosmologies

    pred_avg = np.zeros_like( test_unique )
    pred_err = np.zeros_like( test_unique )             # this will be error on mean (std/sqrt(N))
    count_per_cosmo = np.zeros( test_unique.shape[0] )  # counts the no. of realisations per cosmo

    for i in range( test_unique.shape[0] ):     # scroll through the unique cosmologies
        # Get the indices of rows that match the 1st unique cosmology
        # these are the predictions we need to avg and take the std of.
        idx = np.where( test[:,0] == test_unique[i,0] )[0]
        count_per_cosmo[i] = float( len(idx) )
        for j in range( test.shape[1] ):       # scroll through (Omega_m, S_8, h, w0)
            pred_avg[i,j] = np.mean( pred[idx,j] )
            pred_err[i,j] = np.std( pred[idx,j] ) / np.sqrt(count_per_cosmo[i])
            
    return pred_avg, pred_err, test_unique


def Plot_Accuracy(pred_avg, pred_err, test_unique, xtick_labels, savename):
    # plot the avg accuracies of the predictions and their errors
    fig = plt.figure(figsize = (12,9))
    if pred_avg.shape[1] % 2 ==0:
        # an even number of panels to produce
        nrows = int(pred_avg.shape[1]/2)
    else:
        # odd number of panels, round up number of rows
        nrows = int(pred_avg.shape[1]/2) + 1

    gs1 = gridspec.GridSpec(2, nrows )
    colors = [ 'magenta', 'darkblue', 'dimgrey', 'orange']
    plot_labels = [ r'$\Omega_{\rm m}$', r'$S_8$', r'$h$', r'$w_0$' ]
    for j in range( test_unique.shape[1] ):           # scroll through (Omega_m, S_8, h, w0) 
        ax = plt.subplot(gs1[j], adjustable='box')
        # plot the perfect accuracy line
        ax.plot( np.arange( test_unique.shape[0] )+1, np.zeros( test_unique.shape[0]), 'k-', linewidth=2)
        ax.errorbar( np.arange( test_unique.shape[0] )+1,
                     100.*(pred_avg[:,j]/test_unique[:,j] -1.),
                     yerr=100.*( pred_err[:,j]/test_unique[:,j] ),
                     color=colors[j], fmt='o', markersize=10)
        ax.set_ylabel(r'%s Accuracy' %plot_labels[j] + r' [%]')
        ax.set_xlabel(r'cosmoSLICS ID')
        ax.grid(which='major', axis='both')
        plt.sca(ax)
        plt.xticks(np.arange( test_unique.shape[0] )+1, xtick_labels)
    plt.subplots_adjust(hspace=0)
    plt.savefig( savename )
    #plt.show()
    return

def Plot_Accuracy_vs_Q(Q,pred_avg, pred_err, test_unique, labels, xlabel, savename, ylimits):
    # Plot the accuracy of the CNN vs a quantity Q, e.g., num of layers or filter size.
    # input is array containing numbers of layers
    # and the avg predictions, errors and true values per cosmology.
    fig = plt.figure(figsize = (12,9))
    if pred_avg.shape[1] % 2 ==0:
        # an even number of panels to produce
        nrows = int(pred_avg.shape[1]/2)
    else:
        # odd number of panels, round up number of rows
        nrows = int(pred_avg.shape[1]/2) + 1

    gs1 = gridspec.GridSpec(2, nrows )
    colors = [ 'magenta', 'darkblue', 'lawngreen', 'orange', 'cyan', 'red', 'dimgrey' ]
    plot_labels = [ r'$\Omega_{\rm m}$', r'$S_8$', r'$h$', r'$w_0$' ]
    # Find the value of Q that optimises Acc for each param. 
    Acc_perQ = np.zeros([ test_unique.shape[1], len(Q)  ])   
    opt_Q = np.zeros( test_unique.shape[1] )          # store optimal Q value per cosmol. param.
    
    for j in range( test_unique.shape[1] ):           # scroll through (Omega_m, S_8, h, w0)
        ax = plt.subplot(gs1[j], adjustable='box')
        for i in range(pred_avg.shape[0]):            # scroll through each of the test cosmologies
            # plot the acc. for this test cosmol & param, vs quantity Q
            ax.errorbar( Q,
                         100.*(pred_avg[i,j,:]/test_unique[i,j] -1.),
                         yerr=100.*( pred_err[i,j,:]/test_unique[i,j] ),
                         color=colors[i], linewidth=2,label=r'%s'%labels[i] )
        ax.grid(which='major', axis='both')
        ax.set_ylabel(r'%s Accuracy' %(plot_labels[j])+r' [%]' )
        ax.set_xlabel(xlabel)
        ax.set_ylim(ylimits[j])
        for q in range(len(Q)):
            # calculate the total accuracy across the test cosmols for each value of Q
            Acc_perQ[j,q] = np.sum(  abs((pred_avg[:,j,q]/test_unique[:,j] -1.))  )
        opt_Q[j] = Q[ np.argmin( Acc_perQ[j,:] ) ]
    print("The optimal value of %s per cosmol. param is: "%xlabel, opt_Q)
    ax.legend(loc='best', framealpha=0.5)
    plt.subplots_adjust(hspace=0)
    plt.savefig( savename )
    plt.show()
    return



def Plot_Pred_vs_Q(Q,pred_avg, pred_err, test_unique, labels, xlabel, savename, ylimits):
    # Plot the raw predictions of the CNN, & the true values, vs a quantity Q, e.g., num of layers or filter size.
    # input is array containing numbers of layers
    # and the avg predictions, errors and true values per cosmology.

    fig = plt.figure(figsize = (12,9))
    if pred_avg.shape[1] % 2 ==0:
        # an even number of panels to produce                                                                                      
        nrows = int(pred_avg.shape[1]/2)
    else:
        # odd number of panels, round up number of rows
        nrows = int(pred_avg.shape[1]/2) + 1

    gs1 = gridspec.GridSpec(2, nrows )
    colors = [ 'magenta', 'darkblue', 'lawngreen', 'orange', 'cyan', 'red', 'dimgrey' ]
    plot_labels = [ r'$\Omega_{\rm m}$', r'$S_8$', r'$h$', r'$w_0$' ]

    for j in range( test_unique.shape[1] ):           # scroll through (Omega_m, S_8, h, w0)
        ax = plt.subplot(gs1[j], adjustable='box')
        for i in range(pred_avg.shape[0]):            # scroll through each of the test cosmologies
            # plot the raw pred for this test cosmol & param, vs quantity Q
            ax.plot( [Q[0],Q[-1]], [test_unique[i,j], test_unique[i,j]], linestyle=':', linewidth=2, color=colors[i] )
            ax.errorbar( Q,
                         pred_avg[i,j,:],
                         yerr=pred_err[i,j,:],
                         color=colors[i], linewidth=2,label=r'%s'%labels[i] )
        ax.grid(which='major', axis='both')
        ax.set_ylabel(r'Raw %s' %(plot_labels[j]) )
        ax.set_xlabel(xlabel)
        ax.set_ylim(ylimits[j])
    ax.legend(loc='best', framealpha=0.5)
    plt.subplots_adjust(hspace=0)
    plt.savefig( savename )
    plt.show()
    return
