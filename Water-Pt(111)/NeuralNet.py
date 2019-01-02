from ase.io import *
import numpy as np
import torch
from torch.autograd import Variable
from torch.nn.parameter import Parameter
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.nn.init as init
from sklearn.metrics import mean_squared_error
from math import sqrt
import time
import random
from fingerprint import fingerprint
from sklearn.preprocessing import StandardScaler
import sys
import pickle

#The script belows fits the force components for the H-atoms in dataset.traj. To modify it for fitting O and Pt components, read the comments below. 
#Also, please note that the uploaded dataset dataset.traj only contains the first 1000 images of the full data set described in the report. This is due to file size limitations on GitHub.

############## OPTIONS #############
#Data set sizes
N_train = 500 #number of training images
N_valid = 100 #number of test images
Random = False #choose whether to sample random images from the parent data set or use the images from previous run
batch_factor = 50 #optimization will be done with a batch size of N_train/batch_factor

############# Fingerprint options #########
etas = np.linspace(0.2, 4., 20) #Hyperparameter for fingerprint construction
Rc = 0.*np.ones(len(etas)) #Hyperparameter for fingerprint construction
cutoff = 8. #Only consider atoms within cutoff sphere of each atom
preprocess = True #Scale all fingerprint components to have zero mean and unit variance

#Neural network options
num_epochs = 5000 #Number of epochs to perform optimization
epoch_save = 1 #Choose how often results are saved (every epoch_save'th epoch)
num_hidden = 2 #Number of hidden layers in FFNN
num_l1 = 25 #Number of nodes in first hidden layer
num_l2 = 25 #Number of nodes in second hidden layer
lr = 0.01 #Learning rate for optimizer
weight_decay = 0 #Weight decay when using regularization



################ LOAD TEST AND TRAINING IMAGES ####################

#Training images
if Random == True:
    indices = random.sample(range(1000), N_train+N_valid) #due to file limitations dataset.traj contains only the first 1000 images of the original data set.
    images = []
    for index in indices:
        image = read('../dataset.traj', index = index)
        images.append(image)
    train_images = images[0:N_train]
    valid_images = images[N_train:]
    write('images.traj', images)
    write('train_images.traj', train_images)
    write('valid_images.traj', valid_images)
else:
    images = read('images.traj', index = ':')
    train_images = read('images.traj', index = ':')
    valid_images = read('valid_images.traj', index = ':')



    
############ GET FORCE COMPONENTS #############

#Training images
for count, atoms in enumerate(train_images):
    f = atoms.get_forces()[80:144] #In this case we get the forces on the H atoms. For Pt and O this should be changed.
    f = f.flatten()

    if count == 0:
        forces_train = f
    else:
        forces_train = np.concatenate((forces_train, f), axis = 0)

#Test images
for count, atoms in enumerate(valid_images):
    f = atoms.get_forces()[80:144] #In this case we get the forces on the H atoms. For Pt and O this should be changed.
    f = f.flatten()

    if count == 0:
        forces_valid = f
    else:
        forces_valid = np.concatenate((forces_valid, f), axis = 0) 





########### FINGERPRINTS #####################

fingerprints = fingerprint(images, cutoff, etas, Rc, 'H', cutoff_function = False) #fingerprints are calculated with function from fingerprint.py. 

if preprocess == True:

    ssc = StandardScaler()
    fingerprints = ssc.fit_transform(fingerprints)

fingerprints_train = fingerprints[0:N_train*3*64,:] #The number 64 should be changed to 32 or 48 for the case of O or Pt.
fingerprints_valid = fingerprints[N_train*3*64:] #The number 64 should be changed to 32 or 48 for the case of O or Pt.


        
########### FEED-FORWARD NEURAL NETWORK ################

# define network
class Net(nn.Module):

    def __init__(self, num_features, num_l1, num_l2, num_hidden, num_out):
        super(Net, self).__init__()
        
        # input layer
        self.W_1 = Parameter(init.xavier_normal_(torch.Tensor(num_l1, num_features)))
        self.b_1 = Parameter(init.constant_(torch.Tensor(num_l1), 0))

        if num_hidden == 1:
            
            # hidden layer 1
            self.W_2 = Parameter(init.xavier_normal_(torch.Tensor(num_out, num_l1)))
            self.b_2 = Parameter(init.constant_(torch.Tensor(num_out), 0))

        if num_hidden == 2:
            
            # hidden layer 1
            self.W_2 = Parameter(init.xavier_normal_(torch.Tensor(num_l2, num_l1)))
            self.b_2 = Parameter(init.constant_(torch.Tensor(num_l2), 0))

            # hidden layer 2
            self.W_3 = Parameter(init.xavier_normal_(torch.Tensor(num_out, num_l2)))
            self.b_3 = Parameter(init.constant_(torch.Tensor(num_out), 0))


        # define activation function in constructor
        self.activation = torch.nn.Sigmoid()

    def forward(self, x):
        x = F.linear(x, self.W_1, self.b_1)
        x = self.activation(x)
        x = F.linear(x, self.W_2, self.b_2)
        if num_hidden == 1:
            return x
        else:
            x = self.activation(x)
            x = F.linear(x, self.W_3, self.b_3)
            return x

num_input = fingerprints_train.shape[1] #number of inputs is len(etas)*len(species).
num_out = 1 #There is ONE force component to be output for each input vector.
net = Net(num_input, num_l1, num_l2, num_hidden, num_out)

 
#Set up optimizer and loss function    
optimizer = optim.Adam(net.parameters(), lr = lr, weight_decay = weight_decay)
criterion = nn.MSELoss()




################## TRAINING ###########################
# hyperparameters
batch_size_train = int(len(train_images)*64*3/batch_factor) #The number 64 should be changed to 32 or 48 for the case of O or Pt.
num_samples_train = fingerprints_train.shape[0]
num_batches_train = num_samples_train // batch_size_train
batch_size_valid = int(len(valid_images)*64*3/batch_factor) #The number 64 should be changed to 32 or 48 for the case of O or Pt.
num_samples_valid = fingerprints_valid.shape[0]
num_batches_valid = num_samples_valid // batch_size_valid

# setting up lists for handling loss/accuracy
train_acc, train_loss, train_resid = [], [], []
valid_acc, valid_loss, valid_resid = [], [], []
epochs = []
cur_loss = 0
losses = []


get_slice = lambda i, size: range(i * size, (i + 1) * size)



for epoch in range(num_epochs):
    cur_loss = 0
    net.train()

    for i in range(num_batches_train):
        slce = get_slice(i, batch_size_train)
        fingerprint_batch = Variable(torch.from_numpy(fingerprints_train[slce]).float())
        output = net(fingerprint_batch)[:,0]
        
        # compute gradients given loss
        forces_batch = Variable(torch.from_numpy(forces_train[slce]).float())
        batch_loss = criterion(output, forces_batch)
        optimizer.zero_grad()
        batch_loss.backward()
        optimizer.step()
        
        cur_loss += batch_loss   
    losses.append(cur_loss)

    net.eval()
    
    ### Evaluate training
    train_preds, train_targs = [], []
    for i in range(num_batches_train):
        slce = get_slice(i, batch_size_train)
        fingerprint_batch = Variable(torch.from_numpy(fingerprints_train[slce]).float())        
        forces_batch = Variable(torch.from_numpy(forces_train[slce]).float())
        output = net(fingerprint_batch)
        preds = output[:,0]
        
        train_targs += list(forces_train[slce])
        train_preds += list(preds.data.numpy())

    
    ### Evaluate validation
    val_preds, val_targs = [], []
    for i in range(num_batches_valid):
        slce = get_slice(i, batch_size_valid)
        fingerprint_batch = Variable(torch.from_numpy(fingerprints_valid[slce]).float())        
        output = net(fingerprint_batch)
        preds = output[:,0]
        val_preds += list(preds.data.numpy())
        val_targs += list(forces_valid[slce])

    train_acc_cur = sqrt(mean_squared_error(train_targs, train_preds))
    valid_acc_cur = sqrt(mean_squared_error(val_targs, val_preds))
    train_resid_cur = (np.array(train_targs) - np.array(train_preds)).max()
    valid_resid_cur = (np.array(val_targs) - np.array(val_preds)).max()
    

    if epoch/epoch_save == epoch//epoch_save:   
        print("Epoch %2i : Train Loss %f , Train acc %f, Valid acc %f" % (epoch+1, losses[-1], train_acc_cur, valid_acc_cur))
        epochs.append(epoch)
        train_acc.append(train_acc_cur)
        valid_acc.append(valid_acc_cur)
        train_resid.append(train_resid_cur)
        valid_resid.append(valid_resid_cur)



train_summary = {}
train_summary['epochs'] = epochs
train_summary['train_acc'] = train_acc
train_summary['train_resid'] = train_resid
train_summary['valid_acc'] = valid_acc
train_summary['valid_resid'] = valid_resid
train_summary['val_targs'] = val_targs
train_summary['val_preds'] = val_preds
train_summary['train_targs'] = train_targs
train_summary['train_preds'] = train_preds

with open('train_summary.pickle', 'wb') as f:
    pickle.dump(train_summary, f)



