# 25/11/2020, B. M. Giblin, Postdoc, Edinburgh
# This code contains classes for convolutional neural network analyses

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# find GPU device if available, else use CPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
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
        super(Linear_Net, self).__init__()
        
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


    
class Res_Net(nn.Module):
    # Basically the same as the Linear_Net class, except with a different forward function
    # here, the maps pass through "residual blocks", each consisting of conv+actv+conv
    # before the output is added to the input. For this addition to be possible, the
    # conv's involved in the residual blocks must have sufficient padding so as to not change
    # the size of the maps.
    # The number of residual blocks is controlled by the nclayers argument, just like in Linear_Net.

    def __init__(self, input_channel, output_channel,
                         conv1_filter,conv1_padding,conv1_stride,
                         conv2_filter,conv2_padding,conv2_stride,
                         act1_map_size,num_pCosmol,
                         nclayers, pool_layers):
        super(Res_Net, self).__init__()

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
        

    # ...and this function executes the CNN
    def forward(self, x):
        x = F.relu(self.conv1(x))
        # Put the output map through residual block as many times as nlayers dictates.
        for nl in range(self.nclayers-1):
            y = self.conv2( F.relu(self.conv2(x)) ) # apply residual block 
            x = F.relu( x+y )                       # add output to input & act
            
            if nl in self.pool_layers-2:            # apply pooling after specified res-blocks.
                x = self.pool(x)
        x = x.view(-1, x.shape[1]*x.shape[2]*x.shape[3])      # Last 3 numbers are output dims of previous conv.
        x = self.fc1(x)
        return x


    
class Train_CNN_Class:

    def __init__(self, net, criterion, optimizer):
        self.net=net
        self.criterion=criterion
        self.optimizer=optimizer

    def Train_CNN(self, inputs, labels):
        # zero the parameter gradients
        self.optimizer.zero_grad()

        # forward + backward + optimize
        outputs = self.net(inputs.float())
        loss = self.criterion(outputs.float(), labels.float())
        loss.backward()
        self.optimizer.step()
        return loss

    def Transform_And_Train(self, inputs, labels):
        # Perform 3 90deg rotations & 2 reflections of the minibatches
        # Annoyingly to perform the transformations, you need to convert minibatch to a numpy object

        inputs_np = inputs.detach().cpu().numpy()
        for rot in range(1,4):
            inputs_rot = np.rot90(inputs_np, k=rot, axes=(-2,-1))
            inputs_rot = torch.from_numpy( inputs_rot.copy() ).to(device)
            loss = self.Train_CNN(inputs_rot, labels)

        # First UP/DOWN FLIP
        inputs_ud = torch.from_numpy( inputs_np[:,:,::-1,:].copy() ).to(device) #.double() # up/down flip
        loss = self.Train_CNN(inputs_ud, labels)
        # Second LEFT/RIGHT FLIP
        inputs_lr = torch.from_numpy( inputs_np[:,:,:,::-1].copy() ).to(device) #.double() # left/right flip
        loss = self.Train_CNN(inputs_lr, labels)
        return loss


    
        
class Test_CNN_Class:
    
    def __init__(self, net, Test_Data, Test_Labels, batch_size):
        self.net=net
        self.Test_Data=Test_Data
        self.Test_Labels=Test_Labels
        self.batch_size=batch_size
        # Assemble minibatches & array to store them in.
        self.rand_idx_test = np.arange(0, Test_Data.shape[0])
        self.rand_idx_test = np.reshape( self.rand_idx_test, ( int(len(self.rand_idx_test)/batch_size), batch_size) )
        self.Test_Pred = np.zeros([ Test_Data.shape[0], Test_Labels.shape[1] ])
        
    def Test_CNN(self):
        # Scroll through the minibatches
        for i in range( self.rand_idx_test.shape[0] ):
            inputs = self.Test_Data[ self.rand_idx_test[i] ].to(device)
            labels = self.Test_Labels[ self.rand_idx_test[i] ].to(device)
            outputs = self.net(inputs.float())
            # Store the output predictions
            self.Test_Pred[i*self.batch_size:(i+1)*self.batch_size, :] = outputs.detach().cpu().numpy()
        return self.Test_Pred

    
