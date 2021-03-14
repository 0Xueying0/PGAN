#!/usr/bin/env python
# coding: utf-8

# # Deep Convolutional GANs
# 
# In this notebook, you'll build a GAN using convolutional layers in the generator and discriminator. This is called a Deep Convolutional GAN, or DCGAN for short. The DCGAN architecture was first explored in 2016 and has seen impressive results in generating new images; you can read the [original paper, here](https://arxiv.org/pdf/1511.06434.pdf).
# 
# You'll be training DCGAN on the stock price dataset.
# 
# 
# 
# So, our goal is to create a DCGAN that can generate new, realistic-looking series. We'll go through the following steps to do this:
# * Load in and pre-process the dataset
# * Define discriminator and generator networks
# * Train these adversarial networks
# * Visualize the loss over time and some sample, generated images
# 
# #### Deeper Convolutional Networks
# 
# We'll need a deeper network to accurately identify patterns in these images and be able to generate new ones. Specifically, we'll use a series of convolutional or transpose convolutional layers in the discriminator and generator. It's also necessary to use batch normalization to get these convolutional networks to train. 
# 
# Besides these changes in network structure, training the discriminator and generator networks should be the same as before. That is, the discriminator will alternate training on real and fake (generated) series, and the generator will aim to trick the discriminator into thinking that its generated series are real!

# In[1]:


# import libraries
import matplotlib.pyplot as plt
import tushare as ts
import numpy as np
import pickle as pkl
import torch
from sklearn import preprocessing
from torch.autograd import Variable
import torch.autograd as autograd
import pandas as pd
#%matplotlib inline


# ## Getting the data
# 
# Here you can download the stock dataset. We can load in training data, transform it into Tensor datatypes, then create dataloaders to batch our data into a desired size.

# In[2]:



from torchvision import datasets
from torchvision import transforms


#batch_size = 128
batch_size = 8
num_workers = 0

# build DataLoaders for SVHN dataset

data = pd.read_csv('DJIA.csv',header = 0)
data = data.drop(columns = ['Date'])
data = pd.DataFrame(data.values.T, index=data.columns, columns=data.index)
data = np.array(data)
test = np.diff(data)#return
test = pd.DataFrame(test)
data = pd.DataFrame(data)#price


# In[3]:


#去除price里相等的两列，去除return里的0列
data = data.iloc[:,:-1]
data = data.loc[:, (test != 0).any(axis=0)]
test = test.loc[:, (test != 0).any(axis=0)]
test=test/data#normalised return/price


#test = test.loc[:, (test != 0).any(axis=0)&(test != 1).any(axis=0)]
  # axis=1行，axis=0列


# ### Visualize the Data
# 
# 

# ### Pre-processing: scaling from -1 to 1
# 
# We need to do a bit of pre-processing; we know that the output of our `tanh` activated generator will contain pixel values in a range from -1 to 1, and so, we need to rescale our training images to a range of -1 to 1. (Right now, they are in a range from 0-1.)

# In[4]:



#print("data after diff",test.shape[1])
result = []
time_steps = 60
#print(test.info())
test = test.as_matrix()
for i in range(test.shape[1]-time_steps):
    result.append([test[:,i:i+time_steps]])
result = np.array(result)

originaldata=[]
data = data.as_matrix()
for i in range(data.shape[1]-time_steps):
    originaldata.append([data[:,i:i+time_steps]])
originaldata = np.array(originaldata)   

#print("data after window",result.shape)
#print("data after window",result)
# 训练集和测试集的数据量划分
train_size = int(0.8*result.shape[0])

# 训练集切分

train = result[:train_size,:,:,:]
x_train = train[:,:,:,:-20]
y_train = train[:,:,:,-21:-4]
x_test = result[train_size:,:,:,:-20]
y_test = result[train_size:,:,:,-21:-4]
#x_train = x_train.reshape(x_train.shape[0],x_train.shape[1],x_train.shape[2])
#x_test = x_test.reshape(x_test.shape[0],x_test.shape[1],x_test.shape[2])
#print("x_train", x_train.shape)
#print("y_train", y_train.shape)
#print("x_test", x_test.shape)
#print("y_test", y_test.shape)
#print("x_train one sample",x_train[1,:,:].shape)
#print("y_train one sample",y_train[1,:,:].shape)

data_test = originaldata[train_size:,:,:,:-3]
#print('data_test',data_test.shape)
#print('originaldata',originaldata.shape)
'''
name = ['X_train']
stockdata = pd.DataFrame(columns=name,data = X_train)
stockdata['y_train']=y_train
stockdata.to_csv('./data/day_series_000001.csv')
'''


# ---
# # Define the Model
# 
# A GAN is comprised of two adversarial networks, a discriminator and a generator.

# ## Discriminator
# 
# Here you'll build the discriminator. This is a convolutional classifier like you've built before, only without any maxpooling layers. 
# * The inputs to the discriminator are tensor series
# * You'll want a few convolutional, hidden layers
# * Then a fully connected layer for the output; as before, we want a sigmoid output, but we'll add that in the loss function, [BCEWithLogitsLoss](https://pytorch.org/docs/stable/nn.html#bcewithlogitsloss), later
# 
# 
# For the depths of the convolutional layers I suggest starting with 32 filters in the first layer, then double that depth as you add layers (to 64, 128, etc.). Note that in the DCGAN paper, they did all the downsampling using only strided convolutional layers with no maxpooling layers.
# 
# You'll also want to use batch normalization with [nn.BatchNorm1d](https://pytorch.org/docs/stable/nn.html#batchnorm2d) on each layer **except** the first convolutional layer and final, linear output layer. 
# 
# #### Helper `conv` function 
# 
# In general, each layer should look something like convolution > batch norm > leaky ReLU, and so we'll define a function to put these layers together. This function will create a sequential series of a convolutional + an optional batch norm layer. We'll create these using PyTorch's [Sequential container](https://pytorch.org/docs/stable/nn.html#sequential), which takes in a list of layers and creates layers according to the order that they are passed in to the Sequential constructor.
# 
# Note: It is also suggested that you use a **kernel_size of 5** and a **stride of 2** for strided convolutions.

# In[5]:


def normalization(s):
    a = np.array(s)
    b = (np.max(a)-np.min(a))/np.mean(a)
    return b


# In[6]:


import torch.nn as nn
import torch.nn.functional as F


# helper conv function
def conv(in_channels, out_channels, kernel_size, stride=(1,2), padding=(0,2), batch_norm=False):
    """Creates a convolutional layer, with optional batch normalization.
    """
    layers = []
    conv_layer = nn.Conv2d(in_channels, out_channels, 
                           kernel_size, stride, padding, bias=False)
    
    # append conv layer
    layers.append(conv_layer)

    if batch_norm:
        # append batchnorm layer
        layers.append(nn.BatchNorm1d(out_channels))
     
    # using Sequential container
    return nn.Sequential(*layers)


# In[7]:


class Discriminator(nn.Module):

    def __init__(self, conv_dim=29):
        super(Discriminator, self).__init__()

        # complete init function
        self.conv_dim = conv_dim

        # 32x32 input
        self.conv1 = conv(1, conv_dim*2, (1,5), batch_norm=False) # first layer, no batch_norm
        # 16x16 out
        self.conv2 = conv(conv_dim*2, conv_dim*4, (1,5))
        # 8x8 out
        self.conv3 = conv(conv_dim*4, conv_dim*8, (1,5))
        # 4x4 out
        self.conv4 = conv(conv_dim*8, conv_dim*16, (1,5))
        #
        self.conv5 = conv(conv_dim*16, conv_dim*32, (1,5))
        
        # final, fully-connected layer
        self.fc = nn.Linear(conv_dim*2*conv_dim*32, 1)
#        self.fc = nn.Linear(conv_dim*16, 1)

    def forward(self, x, a):
        
        #print("disriminator input",x.size())
        # all hidden layers + leaky relu activation
        out = F.leaky_relu(self.conv1(x), 0.2)
       # print("discriminator conv1",out.size())
        out = F.leaky_relu(self.conv2(out), 0.2)
      #  print("discriminator conv2",out.size())
        out = F.leaky_relu(self.conv3(out), 0.2)
      #  print("discriminator conv3",out.size())
        out = F.leaky_relu(self.conv4(out), 0.2)
      #  print("discriminator conv4",out.size())
        out = F.leaky_relu(self.conv5(out), 0.2)
      #  print("discriminator conv5",out.size())
#leaky_relu(0.2) 参考Takahashi, S. & Chen, Y. & Tanaka-Ishii, K. (2019). Modeling financial time-series with generative adversarial networks. Physica A: Statistical Mechanics and its Applications. 527. 121261. 10.1016/j.physa.2019.121261.
        #concat with A
#        out = torch.cat((a, out),axis=0)
        # flatten
        out = out.view(-1, self.conv_dim*self.conv_dim*2*32)
#        out = out.view(-1, self.conv_dim*16)
      #  print("discriminator fc input",out.size())
        # final output layer
        out = self.fc(out)
      #  print("discriminator output",out.size())
      #  print("discriminator output",out)
        return out
    


# ## Generator
# 
# Next, you'll build the generator network. The input will be our noise vector `z`, as before. And, the output will be a $tanh$ output, but this time with size 32x32 which is the size of our SVHN images.
# 
# What's new here is we'll use transpose convolutional layers to create our new series. 
# * The first layer is a fully connected layer which is reshaped into a deep and narrow layer. 
# * Then, we use batch normalization and a leaky ReLU activation. 
# * Next is a series of [transpose convolutional layers](https://pytorch.org/docs/stable/nn.html#convtranspose2d), where you typically halve the depth and double the width and height of the previous layer. 
# * And, we'll apply batch normalization and ReLU to all but the last of these hidden layers. Where we will just apply a `tanh` activation.
# 
# #### Helper `deconv` function
# 
# For each of these layers, the general scheme is transpose convolution > batch norm > ReLU, and so we'll define a function to put these layers together. This function will create a sequential series of a transpose convolutional + an optional batch norm layer. We'll create these using PyTorch's Sequential container, which takes in a list of layers and creates layers according to the order that they are passed in to the Sequential constructor.
# 
# Note: It is also suggested that you use a **kernel_size of 5** and a **stride of 2** for transpose convolutions.

# In[8]:


## Size of latent vector to genorator
f_size = 29

# helper deconv function
def deconv(in_channels, out_channels, kernel_size, stride=(1,2), padding=(0,2), batch_norm=False):
    """Creates a transposed-convolutional layer, with optional batch normalization.
    """
    # create a sequence of transpose + optional batch norm layers
    layers = []
    transpose_conv_layer = nn.ConvTranspose2d(in_channels, out_channels, 
                                              kernel_size, stride, padding, bias=False)
    # append transpose convolutional layer
    layers.append(transpose_conv_layer)
    
    if batch_norm:
        # append batchnorm layer
        layers.append(nn.BatchNorm1d(out_channels))
        
    return nn.Sequential(*layers)


# In[9]:


class Generator(nn.Module):
    
    def __init__(self, f_size, conv_dim=29):
        super(Generator, self).__init__()

        # complete init function
        
        self.conv_dim = conv_dim
        
        #conditioning
        self.t_cconv1 = conv(1, conv_dim*2, (1,5))
        self.t_cconv2 = conv(conv_dim*2, conv_dim*2, (1,5))
        self.t_cconv3 = conv(conv_dim*2, conv_dim*2, (1,5))
        self.t_cconv4 = conv(conv_dim*2, conv_dim*2, (1,5))
        
        # conditioning fully-connected layer
        self.fc1 = nn.Linear(conv_dim*2*conv_dim*3, conv_dim)
        
        # simulator fully-connected layer
        self.fc2 = nn.Linear(f_size+conv_dim, 20*conv_dim)
        
        # transpose conv layers
        self.t_sconv1 = deconv(4, 2, (1,5))
        self.t_sconv2 = deconv(2, 1, (1,5))
#        self.t_sconv3 = deconv(conv_dim, 3, 4, batch_norm=False)
        

    def forward(self, x,y):
        
        # hidden transpose conv layers + relu
     #   print(x.size())
        out = F.relu(self.t_cconv1(x))
      #  print("generator cconv1",out.size())
        out = F.relu(self.t_cconv2(out))
      #  print("generator cconv2",out.size())
        out = F.relu(self.t_cconv3(out))
     #   print("generator cconv3",out.size())
        out = F.relu(self.t_cconv4(out))#(1,2,3)
      #  print("generator cconv4",out.size())
        
        # fully-connected + reshape 
        out = out.view(-1, self.conv_dim*self.conv_dim*2*3)
      #  print("generator fc1 input",out.size())
        out = self.fc1(out)
      #  print("generator fc2 output, to be concated",out.size())
#        out = out.view(-1, self.conv_dim*4, 4, 4) # (batch_size, depth, 4, 4)
        
        #concat with A
#        out = torch.cat((A, out),axis=0)

        #concat with latent vector
        out = torch.Tensor(np.concatenate((out.detach().numpy(), y.detach().numpy()), axis=1))
     #   print("generator after concat",out.size())#（1，5）
        
        # fully-connected + reshape 
        out = self.fc2(out)
     #   print("generator fc2",out.size())
#        out=torch.unsqueeze(out,0)
        
        out = out.view(batch_size, 4, self.conv_dim, 5) # (batch_size, depth, 4, 4)
        
        out = F.relu(self.t_sconv1(out))
#       print(out.size())
        out = F.relu(self.t_sconv2(out))
    #    print("generator_out size",out.size())
     #   print("generator_out",out)
        
        # last layer + tanh activation
#        out = self.t_conv3(out)
#        out = F.tanh(out)
        
        return out
    


# ## Build complete network
# 
# Define your models' hyperparameters and instantiate the discriminator and generator from the classes defined above. Make sure you've passed in the correct input arguments.

# In[10]:


# define hyperparams
#number of assets k
#conv_dim = 2
## Size of latent vector to genorator
f_size = 29

# define discriminator and generator
#D = Discriminator(conv_dim)
#G = Generator(f_size=f_size, conv_dim=conv_dim)
D = Discriminator()
G = Generator(f_size=f_size)

print(D)
print()
print(G)


# ### Training on GPU
# 
# Check if you can train on GPU. If you can, set this as a variable and move your models to GPU. 
# > Later, we'll also move any inputs our models and loss functions see (real_images, z, and ground truth labels) to GPU as well.

# In[11]:


train_on_gpu = torch.cuda.is_available()

if train_on_gpu:
    # move models to GPU
    G.cuda()
    D.cuda()
    print('GPU available for training. Models moved to GPU')
else:
    print('Training on CPU.')
    


# ---
# ## Discriminator and Generator Losses
# 
# Now we need to calculate the losses. And this will be exactly the same as before.
# 
# ### Discriminator Losses
# 
# > * For the discriminator, the total loss is the sum of the losses for real and fake images, `d_loss = d_real_loss + d_fake_loss`. 
# * Remember that we want the discriminator to output 1 for real images and 0 for fake images, so we need to set up the losses to reflect that.
# 
# The losses will by binary cross entropy loss with logits, which we can get with [BCEWithLogitsLoss](https://pytorch.org/docs/stable/nn.html#bcewithlogitsloss). This combines a `sigmoid` activation function **and** and binary cross entropy loss in one function.
# 
# For the real images, we want `D(real_images) = 1`. That is, we want the discriminator to classify the the real images with a label = 1, indicating that these are real. The discriminator loss for the fake data is similar. We want `D(fake_images) = 0`, where the fake images are the _generator output_, `fake_images = G(z)`. 
# 
# ### Generator Loss
# 
# The generator loss will look similar only with flipped labels. The generator's goal is to get `D(fake_images) = 1`. In this case, the labels are **flipped** to represent that the generator is trying to fool the discriminator into thinking that the images it generates (fakes) are real!

# In[12]:


def real_loss(D_out, smooth=False):

    batch_size = D_out.size(0)
    if train_on_gpu:
        labels = labels.cuda()
    # calculate loss
    loss = -torch.mean(D_out)
    return loss

def fake_loss(D_out):
    batch_size = D_out.size(0)
    if train_on_gpu:
        labels = labels.cuda()
    # calculate loss
    loss = -torch.mean(D_out)
    return loss


# ## Optimizers
# 
# Not much new here, but notice how I am using a small learning rate and custom parameters for the Adam optimizers, This is based on some research into DCGAN model convergence.
# 
# ### Hyperparameters
# 
# GANs are very sensitive to hyperparameters. A lot of experimentation goes into finding the best hyperparameters such that the generator and discriminator don't overpower each other. Try out your own hyperparameters or read [the DCGAN paper](https://arxiv.org/pdf/1511.06434.pdf) to see what worked for them.

# In[13]:


import torch.optim as optim

# params
lr = 0.00002
beta1=0.5
beta2=0.999 # default value

# Create optimizers for the discriminator and generator
d_optimizer = optim.Adam(D.parameters(), lr, [beta1, beta2])
g_optimizer = optim.Adam(G.parameters(), lr, [beta1, beta2])


# ---
# ## Training
# 
# Training will involve alternating between training the discriminator and the generator. We'll use our functions `real_loss` and `fake_loss` to help us calculate the discriminator losses in all of the following cases.
# 
# ### Discriminator training
# 1. Compute the discriminator loss on real, training images        
# 2. Generate fake images
# 3. Compute the discriminator loss on fake, generated images     
# 4. Add up real and fake loss
# 5. Perform backpropagation + an optimization step to update the discriminator's weights
# 
# ### Generator training
# 1. Generate fake images
# 2. Compute the discriminator loss on fake images, using **flipped** labels!
# 3. Perform backpropagation + an optimization step to update the generator's weights
# 
# #### Saving Samples
# 
# As we train, we'll also print out some loss statistics and save some generated "fake" samples.
# 
# **Evaluation mode**
# 
# Notice that, when we call our generator to create the samples to display, we set our model to evaluation mode: `G.eval()`. That's so the batch normalization layers will use the population statistics rather than the batch statistics (as they do during training), *and* so dropout layers will operate in eval() mode; not turning off any nodes for generating samples.

# In[18]:


lambda_gp = 10
Tensor = torch.FloatTensor

def compute_gradient_penalty(D, real_samples, fake_samples,selfdesign):
    # Random weight term for interpolation between real and fake samples
    alpha = Tensor(np.random.random((real_samples.size(0), 1, 1, 1)))
    # Get random interpolation between real and fake samples
    interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
    d_interpolates = D(x=interpolates,a=selfdesign)
    fake = Variable(Tensor(real_samples.shape[0], 1).fill_(1.0), requires_grad=False)
    # Get gradient w.r.t. interpolates
    gradients = autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=fake,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty

import pickle as pkl
import torch.utils.data as Data        
#torch_dataset = Data.TensorDataset(torch.Tensor([np.array(x_train[:,:,:])]), torch.Tensor([np.array(y_train[:,:,:-2])])) 
torch_dataset = Data.TensorDataset(torch.Tensor(np.array(x_train)), torch.Tensor(np.array(y_train))) 
loader = Data.DataLoader(
    dataset=torch_dataset,      # 数据，封装进Data.TensorDataset()类的数据
    batch_size=8,      # 每块的大小
    shuffle=False,               # 要不要打乱数据 (打乱比较好)
    num_workers=0,              # 多进程（multiprocess）来读数据
)
torch_testset = Data.TensorDataset(torch.Tensor(np.array(x_test)), torch.Tensor(np.array(y_test))) 
loader_test = Data.DataLoader(
    dataset=torch_testset,      # 数据，封装进Data.TensorDataset()类的数据
    batch_size=8,      # 每块的大小
    shuffle=False,               # 要不要打乱数据 (打乱比较好)
    num_workers=0,              # 多进程（multiprocess）来读数据
)
# training hyperparams
num_epochs = 50

# keep track of loss and generated, "fake" samples
samples = []
losses = []

print_every = 100

# Get some fixed data for sampling. These are series that are held
# constant throughout training, and allow us to inspect the model's performance

#five representative simulations
sample_size=5

   

# train the network
#for epoch in range(num_epochs):
for epoch in range(20):    
    for idx, (part_realseries,y_realseries) in enumerate(loader):
        
        real_series = torch.cat((part_realseries,y_realseries),3)
        print("real_series",real_series.shape)
        print("real_series",real_series)
        A = torch.Tensor(np.apply_along_axis(normalization, 1, real_series))
        print("A",A)
        print("A",A.size())

        print("real_series after scale",real_series.size())
        batch_size = real_series.size(0)
        # ============================================
        #            TRAIN THE DISCRIMINATOR
        # ============================================

        d_optimizer.zero_grad()

        # 1. Train with real images

        # Compute the discriminator losses on real images 
        if train_on_gpu:
            real_series = real_series.cuda()
            A=A.cuda()

        D_real = D(real_series,A)

        d_real_loss = real_loss(D_real)

        # 2. Train with fake series

        # Generate fake series

    #normal distribution
        z1 = Variable(torch.Tensor(np.random.normal(0, 1, (batch_size, f_size))))
#        print("z",z1)
#        print("z_shape",z1.size())
        if train_on_gpu:
            z1 = z1.cuda()
            part_realseries = part_realseries.cuda()
        part_fakeseries = G(x=part_realseries,y=z1)
        fake_series = torch.cat((part_realseries,part_fakeseries),3)
#        print("fake_series1",fake_series1.size())
        
        # Compute the discriminator losses on fake series            
        D_fake = D(fake_series,A)
        d_fake_loss = fake_loss(D_fake)
        gradient_penalty = compute_gradient_penalty(D, real_series, fake_series,A)
        # add up loss and perform backprop
        d_loss = d_real_loss - d_fake_loss + lambda_gp * gradient_penalty
        d_loss.backward()
        d_optimizer.step()


        # =========================================
        #            TRAIN THE GENERATOR
        # =========================================
        g_optimizer.zero_grad()

        # 1. Train with fake images and flipped labels

        # Generate fake series
    #        z = np.random.uniform(-1, 1, size=(batch_size, z_size))
    #        z = torch.from_numpy(z).float()
    #normal distribution
        z1 = Variable(torch.Tensor(np.random.normal(0, 1, (batch_size, f_size))))

        if train_on_gpu:
            z1 = z1.cuda()
        part_fakeseries = G(x=part_realseries,y=z1)
        fake_series = torch.cat((part_realseries,part_fakeseries),3)

        # Compute the discriminator losses on fake images 
        # using flipped labels!
        D_fake = D(fake_series,A)
        g_loss = real_loss(D_fake) # use real loss to flip labels

        # perform backprop
        g_loss.backward()
        g_optimizer.step()

        # Print some loss stats
        if idx % print_every == 0:
            # append discriminator loss and generator loss
            losses.append((d_loss.item(), g_loss.item()))
            # print discriminator and generator loss
            print('Epoch [{:5d}/{:5d}] | d_loss: {:6.4f} | g_loss: {:6.4f}'.format(
                    epoch+1, num_epochs, d_loss.item(), g_loss.item()))

    
    ## AFTER EACH EPOCH##    
    # generate and save sample, fake images
    G.eval() # for generating samples
    if train_on_gpu:
        fixed_z = fixed_z.cuda()
        part_realseries = part_realseries.cuda()
    fixed_z = Variable(torch.Tensor(np.random.normal(0, 1, (batch_size, f_size))))   
    samples_z = G(x=torch.Tensor(x_test[:batch_size,:,:,:]),y=fixed_z)
    samples.append(samples_z)
#    print("samples",samples)    
    G.train() # back to training mode
'''
if train_on_gpu:
    fixed_z = fixed_z.cuda()
for idx, (part_realseries,y_realseries) in enumerate(loader_test):
    if train_on_gpu:
        part_realseries = part_realseries.cuda()
    samples_z = G(x=part_realseries,y=fixed_z)
    samples.append(samples_z)
    '''
# Save training generator samples
with open('train_samples.pkl', 'wb') as f:
    pkl.dump(samples, f)


# ## Training loss
# 
# Here we'll plot the training losses for the generator and discriminator, recorded after each epoch.

# In[19]:


fig, ax = plt.subplots()
losses = np.array(losses)
plt.plot(losses.T[0], label='Discriminator', alpha=0.5)
plt.plot(losses.T[1], label='Generator', alpha=0.5)
plt.title("Training Losses")
plt.legend()


# In[20]:


t = np.concatenate((x_test[0,0,0,:],np.array(samples_z.detach().numpy())[0,0,0,:]),0)
t = t*data_test[0,0,0,:]+data_test[0,0,0,:]
fig, ax = plt.subplots()
plt.plot(t, label='test', alpha=0.5)
#plt.plot(samples, label='Generator', alpha=0.5)
plt.title("AAPL.O")
plt.legend()


# In[21]:


x_test[:batch_size,:,:,:].shape


# In[2]:


fixed_z1 = Variable(torch.Tensor(np.random.normal(0, 1, (batch_size, f_size))))
samples_z = G(x=torch.Tensor(x_test[:batch_size,:,:,:]),y=fixed_z1)
for j in range(29):
    title=['AAPL.O','AXP.N','BA.N','CAT.N','CSCO.O','CVX.N','DIS.N','GS.N','HD.N','IBM.N','INTC.O','JNJ.N','JPM.N','KO.N','MCD.N','MMM.N','MRK.N','MSFT.O','NKE.N','PFE.N','PG.N','TRV.N','UNH.N','UTX.N','V.N','VZ.N','WBA.O','WMT.N','XOM.N']
    t = np.concatenate((x_test[0,0,j,:],np.array(samples_z.detach().numpy())[0,0,j,:]),0)
    p = t*data_test[0,0,j,:]+data_test[0,0,j,:]
    real = data_test[0,0,j,:]
    fig, ax = plt.subplots()
    x=list(range(57))
    plt.plot(x[:41],t[:41], label='real',alpha=0.5)
    plt.plot(x[40:],t[40:], label='simulation',alpha=0.5)
    plt.plot(x[40:],y_test[0,0,j,:], label='real',alpha=0.5)
#    plt.plot(x[:41],p[:41], label='real data', alpha=0.5)
#    plt.plot(x[40:],p[40:], label='simulation data',alpha=0.5)
#    plt.plot(x[40:],real[40:], label='real',alpha=0.5)
    #plt.plot(samples, label='Generator', alpha=0.5)
    plt.title(title[j])
    plt.legend()


# In[ ]:


fixed_z1 = Variable(torch.Tensor(np.random.normal(0, 1, (batch_size, f_size))))
samples_z = G(x=torch.Tensor(x_test[:batch_size,:,:,:]),y=fixed_z1)
for j in range(29):
    title=['AAPL.O','AXP.N','BA.N','CAT.N','CSCO.O','CVX.N','DIS.N','GS.N','HD.N','IBM.N','INTC.O','JNJ.N','JPM.N','KO.N','MCD.N','MMM.N','MRK.N','MSFT.O','NKE.N','PFE.N','PG.N','TRV.N','UNH.N','UTX.N','V.N','VZ.N','WBA.O','WMT.N','XOM.N']
    t = np.concatenate((x_test[0,0,j,:],np.array(samples_z.detach().numpy())[0,0,j,:]),0)
    p = t*data_test[0,0,j,:]+data_test[0,0,j,:]
    real = data_test[0,0,j,:]
    fig, ax = plt.subplots()
    x=list(range(57))

    plt.plot(x[:41],p[:41], label='real data', alpha=0.5)
    plt.plot(x[40:],p[40:], label='simulation data',alpha=0.5)
    plt.plot(x[40:],real[40:], label='real',alpha=0.5)
    #plt.plot(samples, label='Generator', alpha=0.5)
    plt.title(title[j])
    plt.legend()


# ## Generator samples from training
# 
# Here we can view samples of images from the generator. We'll look at the images we saved during training.

# In[ ]:




