# 24/11/2020, B. M. Giblin, Postdoc, Edinburgh
# Read in a saved CNN and test its predictions for KiDS1000

import numpy as np                                # Useful for managing/reading data
from astropy.io import fits  
import sys
from Functions_4_CNN import Slow_Read, Transform_Data, Untransform_Data, Avg_Pred, Plot_Accuracy, Plot_Accuracy_vs_Q
import torch          # main neural net module
import torch.nn as nn
import torch.nn.functional as F
# find GPU device if available, else use CPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Establish the parameters of the mocks used to train the CNN

mock_Type = "KiDS1000"
if mock_Type == "KV450":
    Realisations = 25
elif mock_Type== "KiDS1000":
    Realisations = 10

Data_Type = "Shear"           # Ultimately may use kappa maps as well
                              # in which case this variable will change.
                              # !!! Note, still need to code in this functionality !!!

ZBlabel = ['ZBcut0.1-0.3', 'ZBcut0.3-0.5', 'ZBcut0.5-0.7', 'ZBcut0.7-0.9', 'ZBcut0.9-1.2']
          #['ZBcut0.1-1.2']   # The redshift range imposed on the maps
          
Augment_Train = True          # If True, within the training loop, it will perform rotations/reflections
                              # of each map, to help the CNN learn cosmol. is invariant to these.
Augment_Data = False          # Leave False. If True, it reads rot'd/ref'd maps into memory.
                              # This is redundant if Augment_Train is True, and former takes less memory.

Noise = "On"                # The shape noise level in the maps: "None"/"On"

Res = 128                     # The number of pxls on each side of the maps
nclayers = int(sys.argv[1])   # The number of conv. layers to have in the CNN
                              # The number of fully connected (FC) layers is currently fixed to 1.
conv1_filter = int(sys.argv[2])

print("Setting up a CNN with %s conv layers." %nclayers)

RUN = int(sys.argv[3])        # Variable for checking convergence of CNN predictions
                              # run the CNN several times changing only this variable
                              # and compare output to verify convergence.                                                    

