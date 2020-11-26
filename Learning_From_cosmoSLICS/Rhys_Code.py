#imports for data handling
import matplotlib.pyplot as plt
import os
import re
import pandas as pd
import xarray as xr
from pathlib import PurePath as PP
import time
import sys
import numpy as np
import matplotlib.gridspec as gridspec
from astropy.io import fits
from astropy.visualization import astropy_mpl_style
plt.style.use(astropy_mpl_style)

#imports for the CNN
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import random
import torch.optim as optim

#DATA SPECS
CS_DIR = r'cosmoSLICS/'          # cosmoSLICS directory  
mock_Type = 'KV450'
Data_Type = "Shear"
Augment = False
RUN=0
#ZBlabel = ['ZBcut0.1-1.2'] 
ZBlabel = ['ZBcut0.1-0.3', 'ZBcut0.3-0.5', 'ZBcut0.5-0.7', 'ZBcut0.7-0.9', 'ZBcut0.9-1.2']# The redshift range imposed on the maps
Noise = 'None'           # The shape noise level in the maps
Res = 128                # The number of pxls on each side of the maps                                                                                                    
Realisations = 25
Quads = 4
seeds = 2

Num_Layers = 2
num_cos_params = 4
batch = 4

#DATA SPLITTING
Split_Type = "Fid"  

if Split_Type == "Fid":
    Test_mockIDs = ['00','05','07','12','18','fid']     # the test mock IDs
    Test_LOS = 12

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

#CHECK/MAKE DIRS                                                                                                                                                                                  
read_directory = 'QuickReadData/SplitType-%s' %Split_Type
if not os.path.exists(read_directory):
    os.makedirs(read_directory)  

def make_outdir():
    Out_DIR = 'Results_CNN/Mock%s_Data%s_Split%s_zBins%s_Noise%s_Aug%s_Res%s/Net_convlayers%s_FClayers1'\
    %(mock_Type,Data_Type,Split_Type,len(ZBlabel),Noise,Augment,Res,Num_Layers)
    if not os.path.exists(Out_DIR):
        os.makedirs(Out_DIR)
    return Out_DIR

#READ DATA
def rhysread(IDs,kind):

    try:
        Shear = np.load("%s/Rhys_%s_Shear_z%s_SN%s_Res%s.npy" %(read_directory,kind,len(ZBlabel),Noise, Res))
        names = np.load("%s/Rhys_%s_names_z%s_SN%s_Res%s.npy" %(read_directory,kind,len(ZBlabel),Noise, Res))
        print("Loaded from file!")

    except:

        cosmols = ["\\" + i if kind == "train" else "foobar" for i in IDs]

        LOS = ["LOS"+str(i+1)+"_" for i in range(Realisations)]
        newLOS = LOS
        [newLOS.remove("LOS"+str(i+1)+"_") for i in range(Test_LOS) if kind == "train"]
        [newLOS.remove("LOS"+str(i+13)+"_") for i in range(Realisations - Test_LOS) if kind == "test"]
        extra = ["\\" + i for i in Test_mockIDs]

        t1 = time.time()
        names1,names2 = [[] for i in range(len(ZBlabel))],[[] for i in range(len(ZBlabel))]
        maps1,maps2 = [[] for i in range(len(ZBlabel))],[[] for i in range(len(ZBlabel))]
            
        #walk through all directories
        for path, subdirs, files in os.walk(CS_DIR):
            for name in files:
                if "Kappa" in name or "xipm" in name: #ignore non shear maps
                    continue
                elif "Quad" in name: #only take quadrant files
                    absp = PP(path, name)
                    relp = os.path.relpath(absp, str(os.getcwd()))
                    ZB = re.search('Res128x128(.+?)Shear', str(relp)).group(1)
                    ZBox = ZB.replace("\\","") #get redshift of current file
                    if (any([x in str(relp) for x in cosmols])\
                        or (any([x in str(relp) for x in extra])\
                        and any([x in str(name) for x in newLOS])))\
                        and any([x in str(relp) for x in ZBlabel]):

                            if "Shear1" in name:
                                names1[ZBlabel.index(ZBox)].append(relp)
                                f = fits.getdata(relp, ext=0)
                                maps1[ZBlabel.index(ZBox)].append(f)

                            if "Shear2" in name:
                                names2[ZBlabel.index(ZBox)].append(relp)
                                f = fits.getdata(relp, ext=0)
                                maps2[ZBlabel.index(ZBox)].append(f)

        t2 = time.time()                               # Take another timer measurement
        print("Rhys: Time taken to read in maps is %.1f seconds." %(t2-t1))

        Shear = np.zeros([ len(maps1[0]), 2*len(maps1),128,128 ])

        for a,(i,j) in enumerate(zip(maps1,maps2)):
            Shear[:,a,:,:]=i
            Shear[:,a+len(maps1),:,:]=j
        

        print(kind + " set dims: " + str(Shear.shape))
        names = np.concatenate((names1,names2))

        np.save("%s/Rhys_%s_Shear_z%s_SN%s_Res%s" %(read_directory,kind,len(ZBlabel),Noise, Res), Shear)
        np.save("%s/Rhys_%s_names_z%s_SN%s_Res%s" %(read_directory,kind,len(ZBlabel),Noise, Res), names)

    return Shear, names


def rhysout(names,IDs,kind):
    extras = (Realisations - Test_LOS) * len(Test_mockIDs) * seeds * Quads
    extra = extras if kind == "train" else -extras
    Cosmols = np.loadtxt(CS_DIR+'/cosmoSLICS_Cosmologies_Omm-S8-h-w0-sig8-sig8bf.dat')[:,0:4] # Only take the frst 4 columns
    Cosmols_Stack = np.zeros([len(IDs)*seeds*Realisations*Quads + extra, 4 ])

    ID = ["\\" + i for i in mockIDs]

    idx = []

    for a,i in enumerate(names[0,:]):
        for j in ID:
            if str(j) in i:
                b = int(j.replace("\\fid", "25")) if "fid" in j else int(j.replace("\\",""))
                Cosmols_Stack[a,:]=Cosmols[b,:]
                idx.append('%.2d' %b)

    Cosmols_df = pd.DataFrame(Cosmols_Stack, index=idx, columns = ["Omm","S8", "h", "W0"])

    return Cosmols_df


ShearTe, namesTe = rhysread(IDs=Test_mockIDs, kind="test")
ShearTr, namesTr = rhysread(IDs=Train_mockIDs, kind="train")

outputTe = rhysout(namesTe,Test_mockIDs, kind="test")
outputTr = rhysout(namesTr,Train_mockIDs, kind="train")


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#device = torch.device("cpu")
# Assuming that we are on a CUDA machine, this should print a CUDA device:

def transform(data, mean, std):
    trans = np.zeros_like(data)
    for i in range( len(mean) ):
        trans[:,i,:,:] = (data[:,i,:,:] - mean[i]) / std[i]
    return trans

def untransform(data, mean, std):
    untrans = np.zeros_like(data)
    for i in range( len(mean) ):
        untrans[:,i,:,:] = std[i]*data[:,i,:,:] + mean[i]
    return untrans

# A mean and stan-dev for each channel - used to normalise data
Mean_T = np.zeros(ShearTr.shape[1])
Std_T = np.zeros(ShearTr.shape[1])
for i in range(ShearTr.shape[1]):
    Mean_T[i] = np.mean( ShearTr[:,i,:,:] )
    Std_T[i] = np.std( ShearTr[:,i,:,:] )

# Convert inputs to torch tensors
outputTr_Torch = torch.from_numpy( outputTr.to_numpy() ) 
outputTe_Torch = torch.from_numpy( outputTe.to_numpy() )

ShearTr = torch.from_numpy( transform(ShearTr, Mean_T, Std_T) ).double()
ShearTe = torch.from_numpy( transform(ShearTe, Mean_T, Std_T) ).double()

#DEFINE NN VARIABLES
conv1_pad = 0
conv1_f = 5
conv1_stride = 1

act1_map_sz = int((1+Res - conv1_f + 2*conv1_pad) / conv1_stride)

conv2_pad = 2
conv2_f = conv1_f
conv2_stride = conv1_stride
out_chans = 128

#Architecture Keyword
Arch_keyname = 'conv1F%sP%sS%s-conv2F%sP%sS%s_FC%sCosmol_run%s'\
    %(conv1_f,conv1_pad,conv1_stride,conv2_f,conv2_pad,conv2_stride,num_cos_params, RUN)


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(2*len(ZBlabel),out_chans,conv1_f,conv1_stride,conv1_pad)
        self.conv2 = nn.Conv2d(out_chans, out_chans,conv2_f,conv2_stride,conv2_pad)
        self.fc = nn.Linear(out_chans*act1_map_sz*act1_map_sz, num_cos_params)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        for No_L in range(1,Num_Layers):
            x = F.relu(self.conv2(x))
        x = x.view(-1, x.shape[1]*x.shape[2]*x.shape[3])
        x = self.fc(x)
        return x

def train(optimizer, net, save = True, load = False):
    Out_DIR = make_outdir()

    PATH = '%s/Net_%s.pth' %(Out_DIR,Arch_keyname)
    t1 = time.time()
    losslist = []
    minibatches = []

    if load == False:
        for epoch in range(1):  # loop over the dataset multiple times

            running_loss = 0.0

            indices = [i for i in range(ShearTr.shape[0])]
            random.shuffle(indices)
            indices = np.reshape( np.array(indices), ( int(len(indices)/batch), batch) )

            for i in range(indices.shape[0]):
                # get the inputs; data is a list of [inputs, labels]
                inputs = ShearTr[indices[i]].to(device).float()
                labels = outputTr_Torch[indices[i]].to(device).float()

                if Augment == True:
                    for j in range(6):
                        #Apply transformation
                        inputs_np = inputs.detach().cpu().numpy() # turn minibatch into a numpy object
                        if j < 4:
                            inputs_t = torch.from_numpy(np.rot90(inputs_np, j, (-2,-1)).copy()).to(device)
                        elif j==4:
                            inputs_t = torch.from_numpy(np.flip(inputs_np,-2).copy()).to(device)
                        elif j==5:
                            inputs_t = torch.from_numpy(np.flip(inputs_np,-1).copy()).to(device)

                        # zero the parameter gradients
                        optimizer.zero_grad()

                        # forward + backward + optimize
                        outputs = net(inputs_t).float()
                        loss = criterion(outputs.float(), labels.float())
                        loss.backward()
                        optimizer.step()

                        # print statistics
                        running_loss += loss.item()
                        if i % 50 == 49:    # print every 10 mini-batches
                            print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / 10))
                            losslist.append(running_loss)
                            minibatches.append(i+epoch*(1150))
                            running_loss = 0.0

                else:
                    # zero the parameter gradients
                    optimizer.zero_grad()

                    # forward + backward + optimize
                    outputs = net(inputs).float()
                    loss = criterion(outputs.float(), labels.float())
                    loss.backward()
                    optimizer.step()

                    # print statistics
                    running_loss += loss.item()
                    if i % 50 == 49:    # print every 10 mini-batches
                        print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / 10))
                        losslist.append(running_loss)
                        minibatches.append(i+epoch*(1150))
                        running_loss = 0.0

        t2 = time.time()
        print('Finished Training. Took %.1f seconds.' %(t2-t1))

    else:
        net = Net()
        net.to(device).float()
        net.load_state_dict(torch.load(PATH))
        print("NN loaded from file!")
        losslist = []
        minibatches = []

    if save == True:
        torch.save(net.state_dict(), PATH)

    return net, losslist, minibatches

def make_predictions(ShearTe, outputTe_Torch, outputTe, batch, net):
    #Now feed in Test sets in the same format

    indices = [i for i in range(ShearTe.shape[0])]
    indices = np.reshape( np.array(indices), ( int(len(indices)/batch), batch) )

    #prepare array for net predictions
    predictions = np.zeros([ ShearTe.shape[0], outputTe_Torch.shape[1] ])
    for i in range(indices.shape[0]):
        inputs = ShearTe[indices[i]].to(device).float()
        labels = outputTe_Torch[indices[i]].to(device).float()
        outputs = net(inputs).float()
        # Store the output predictions
        predictions[i*batch:(i+1)*batch, :] = outputs.cpu().detach().numpy()

    predictions = pd.DataFrame(predictions, index = outputTe.index,  columns = outputTe.columns)
    print(predictions)
    return predictions

def average_preds(pred,true):
    Out_DIR = make_outdir()
    #Take means and STDs for each unique cosmology
    means = pd.DataFrame()
    stds = pd.DataFrame()
    for ind in Test_mockIDs:
        indx = ind.replace("fid", "25")
        indices = [indx]
        frame = pred.loc[indx,:]
        
        mean = pd.DataFrame([frame.mean()],index=indices)
        std = pd.DataFrame([frame.std()],index=indices)

        means = means.append(mean)
        stds = stds.append(std)

    #get rid of duplicates in true df
    true2 = true.drop_duplicates(inplace = False)
    # Save output predictions:
    means.to_pickle('%s/TestPredAvg_%s' %(Out_DIR,Arch_keyname))
    stds.to_pickle('%s/TestPredErr_%s' %(Out_DIR,Arch_keyname))
    true2.to_pickle('%s/TestTrue'%Out_DIR)

    return means, stds, true2, frame.size

def Accuracy(means, stds, true, data_length, savename):
    # Now plot the accuracies of the predictions
    fig = plt.figure(figsize = (8,6))
    gs1 = gridspec.GridSpec(2, 2)
    colors = [ 'magenta', 'darkblue', 'dimgrey', 'orange']
    plot_labels = [ r'$\Omega_{\rm m}$', r'$S_8$', r'$h$', r'$w_0$' ]

    for i, j in enumerate(outputTe.columns):           # scroll through (Omega_m, S_8, h, w0) 
        ax = plt.subplot(gs1[i], adjustable='box')
        ax.errorbar( np.arange( means[j].shape[0] )+1, 100.*(means[j]/true[j] -1.),
                     yerr=100.*( stds[j]/true[j]/np.sqrt(data_length) ),
                     color=colors[i] )
        ax.set_ylabel(r'%s Accuracy' %plot_labels[i] + r' [\%]')
        ax.set_xlabel(r'cosmoSLICS ID')
    plt.tight_layout()
    plt.savefig( savename )
    plt.show()

def Accuracy_vs_Layers(num_layers, means, stds, true, data_length, savename):
    # Now plot the accuracies of the predictions
    fig = plt.figure(figsize = (12,9))
    nrows = 2#int((means.shape[1]+1)/2)
    gs1 = gridspec.GridSpec(2, nrows)
    colors = [ 'magenta', 'darkblue', 'lawngreen', 'orange', 'cyan', 'red', 'dimgrey' ]
    plot_labels = [ r'$\Omega_{\rm m}$', r'$S_8$', r'$h$', r'$w_0$' ]

    for i, param in enumerate(outputTe.columns):           # scroll through (Omega_m, S_8, h, w0) 
        ax = plt.subplot(gs1[i], adjustable='box')
        for j, cos in enumerate(Test_mockIDs):
            cosm = cos.replace("fid", "25")
            ax.errorbar( num_layers, 100.*(means.sel(index=cosm)[param]/true[param][cosm] -1.),
                        yerr=100.*( stds.sel(index=cosm)[param]/true[param][cosm]/np.sqrt(data_length) ),
                        color=colors[j], label=cos )
        ax.set_ylabel(r'%s Accuracy' %plot_labels[i] + r' [\%]')
        ax.set_xlabel(r'Nr of Layers')
        #ax.set_yscale("log")
        ax.set_ylim(-50,50)

    ax.legend()
    plt.tight_layout()
    plt.savefig( savename )
    plt.show()

Run_Single = False
if Run_Single:
    Out_DIR=make_outdir()
    net = Net()
    net.to(device).float() #put net on GPU

    criterion = nn.MSELoss()
    optimizer = optim.Adam(net.parameters(), lr=0.0001)

    net, losslist, minibatches = train(optimizer=optimizer, net=net, load=False)
    predictions = make_predictions(ShearTe, outputTe_Torch, outputTe, batch, net)
    means, stds, true, data_length = average_preds(predictions, outputTe)
    savename = '%s/PlotAcc_%s.png' %(Out_DIR, Arch_keyname)
    Accuracy(means,stds,true,data_length,savename)

Run_Many = False
if Run_Many:
    layers = [1,2,3,4,5,6,7,8,9,10]
    #layers = [1]
    batch = 4

    for i in layers:
        Num_Layers = i

        net = Net()
        net.to(device).float() #put net on GPU

        criterion = nn.MSELoss()
        optimizer = optim.Adam(net.parameters(), lr=0.0001)

        net, losslist, minibatches = train(optimizer=optimizer, net=net, load=False)
        predictions = make_predictions(ShearTe, outputTe_Torch, outputTe, batch, net)
        means, stds, true, data_length = average_preds(predictions, outputTe)
        #savename = '%s/PlotAcc_%s.png' %(Out_DIR, Arch_keyname)
        #Accuracy(means,stds,true,data_length,savename)

        plt.plot(minibatches,losslist,label = 'Layers: %s'%i)

    plt.yscale("log")
    plt.legend()
    plt.show()

Load_Many = True
if Load_Many:
    #Now load in the DFs
    Layers = np.arange(1,11)
    means_stack, stds_stack = [], []
    for i in Layers:
        Out_DIR = make_outdir()
        temp_Out_DIR = Out_DIR.split('convlayers')[0] + 'convlayers%s_FC'%i + Out_DIR.split('FC')[-1]
        means = pd.read_pickle('%s/TestPredAvg_%s' %(temp_Out_DIR,Arch_keyname))
        print(means)
        means_stack.append(means.to_xarray())
        stds = pd.read_pickle('%s/TestPredErr_%s' %(temp_Out_DIR,Arch_keyname))
        stds_stack.append(stds.to_xarray())
    multi_savename = '%s/Plot-Nclayers%s-%s_%s.png' %(temp_Out_DIR, Layers[0], Layers[-1], Arch_keyname)
    means_stacked = xr.concat(means_stack, dim = [str(i) for i in Layers])
    stds_stacked = xr.concat(stds_stack, dim = [str(i) for i in Layers])
    true = pd.read_pickle('%s/TestTrue'%Out_DIR)

print(data_length-384)
#data_length = 384
Accuracy_vs_Layers(Layers,means_stacked,stds_stacked,true,data_length,multi_savename)