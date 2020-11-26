# 25/11/2020, B. M. Giblin, Postdoc, Edinburgh
# This code contains classes for convolutional neural network analyses

import numpy as np
import torch.nn as nn
import torch.nn.functional as F

        
class Linear_Net(nn.Module):
    
    # This CNN is designed to apply one unique initial conv,
    # defined by a filter size, padding and stride,
    # and then for all subsequent convolutions to have a common
    # value of filter size, padding and stride, which may differ from the 1st conv.
    # The number of output channels for all convolutions is fixed to output_channel.
    # The number of subsequent convolutions is determined by nclayers.
    # Whether pooling is applied, and after which conv layers, is set
    # by the pool_layers array.
    # Finally, the fully_connected layer converts the final actv map,
    # which contains output_channel x act1_map_size^2 elements, into num_pCosmol numbers.
    # act1_map_size, the dimensions of the final output actv map, must be calculated in
    # advance for the given achitecture, using the function Calc_Output_Map_Size in
    # Functions_4_CNN.py 
    
    # This bit of code defines the architecture of the CNN
    def __init__(self, input_channel, output_channel,
                         conv1_filter,conv1_padding,conv1_stride,
                         conv2_filter,conv2_padding,conv2_stride,
                         act1_map_size,num_pCosmol,
                         nclayers, pool_layers):
        super(Net, self).__init__()
        
        # Parameters defining the architecture
        self.input_channel = input_channel
        self.output_channel = output_channel
        self.conv1_filter = conv1_filter
        self.conv1_padding = conv1_padding
        self.conv1_stride = conv1_stride
        self.conv2_filter = conv2_filter
        self.conv2_padding = conv2_padding
        self.conv2_stride = conv2_stride
        self.act1_map_size = act1_map_size
        self.num_pCosmol = num_pCosmol
        self.nclayers = nclayers
        self.pool_layers = pool_layers
        
        self.pool = nn.MaxPool2d(2, 2)
        # Defines the first convolution:
        self.conv1 = nn.Conv2d(self.input_channel, self.output_channel, self.conv1_filter,
                               padding=self.conv1_padding, stride=self.conv1_stride)
        self.conv2 = nn.Conv2d(self.output_channel, self.output_channel, self.conv2_filter,
                               padding=self.conv2_padding, stride=self.conv2_stride)
        self.fc1 = nn.Linear(self.output_channel*self.act1_map_size*self.act1_map_size, self.num_pCosmol)
                               # act1_map_size is the num pxls on each side of the actv map output from conv layer 1
                               # and we have chosen the padding on conv2 to keep this unchanged,
                               # hence output_channel*act1_map_size^2 is the dimensions of the activ. map from conv2

    # ...and this function executes the CNN
    def forward(self, x):
        x = F.relu(self.conv1(x))
        # Put the output map through conv2 as many times as nlayers dictates.
        for nl in range(self.nclayers-1):
            x = F.relu(self.conv2(x))
            if nl in self.pool_layers-2:
                x = self.pool(x)
        x = x.view(-1, x.shape[1]*x.shape[2]*x.shape[3])      # Last 3 numbers are output dims of previous conv.
        x = self.fc1(x)
        return x


        
