############################################
#   Prediction
############################################

import numpy as np
import torch

from pn2v.utils import imgToTensor
from pn2v.utils import denormalize
from pn2v.utils import normalize


def predict(im, net, noiseModel, device):
    '''
    Process an image using our network.
    
    Parameters
    ----------
    im: numpy array
        2D image we want to process
    net: a pytorch model
        the network we want to use
    noiseModel: NoiseModel
        The noise model to be used.
    device:
        The device your network lives on, e.g. your GPU
        
    Returns
    ----------
    means: numpy array
        Image containing the means of the predicted prior distribution.
        This is similar to normal N2V.
    mseEst: numpy array
        Image containing the MMSE prediction, computed using the prior and noise model.
    '''
    stdTorch=torch.Tensor(np.array(net.std)).to(device)
    meanTorch=torch.Tensor(np.array(net.mean)).to(device)
    
    #im=(im-net.mean)/net.std
    
    inputs_raw= torch.zeros(1,1,im.shape[0],im.shape[1])
    inputs_raw[0,:,:,:]=imgToTensor(im);

    # copy to GPU
    inputs_raw = inputs_raw.to(device)
    
    # normalize
    inputs = (inputs_raw-meanTorch)/stdTorch

    output=net(inputs)

    
    samples = (output).permute(1, 0, 2, 3)*10 #We found that this factor can speed up training
    
    # denormalize
    samples = samples * stdTorch + meanTorch

    # call likelihood using denormalized observations and samples
    likelihoods=noiseModel.likelihood(inputs_raw ,samples )


    mseEst = torch.sum(likelihoods*samples,dim=0,keepdim=True)[0,...] # Sum up over all samples
    mseEst/= torch.sum(likelihoods,dim=0,keepdim=True)[0,...] # Normalize

    means = torch.mean(samples,dim=0,keepdim=True)[0,...] # Sum up over all samples

    # Get data from GPU
    mseEst=mseEst.cpu().detach().numpy()
    means=means.cpu().detach().numpy()

    # Reshape to 2D images and remove padding
    means.shape=(output.shape[2],output.shape[3])


    mseEst.shape=(output.shape[2],output.shape[3])


    return means,mseEst



def tiledPredict(im, net, ps, overlap, noiseModel, device):
    '''
    Tile the image to save GPU memory.
    Process it using our network.
    
    Parameters
    ----------
    im: numpy array
        2D image we want to process
    net: a pytorch model
        the network we want to use
    ps: int
        the widht/height of the square tiles we want to use in pixels
    overlap: int
        number of pixels we want the tiles to overlab in x and y
    noiseModel: NoiseModel
        The noise model to be used.
    device:
        The device your network lives on, e.g. your GPU
        
    Returns
    ----------
    means: numpy array
        Image containing the means of the predicted prior distribution.
        This is similar to normal N2V.
    mseEst: numpy array
        Image containing the MMSE prediction, computed using the prior and noise model.
    '''
    
    means=np.zeros(im.shape)
    mseEst=np.zeros(im.shape)
    xmin=0
    ymin=0
    xmax=ps
    ymax=ps
    ovLeft=0
    while (xmin<im.shape[1]):
        ovTop=0
        while (ymin<im.shape[0]):
            a,b = predict(im[ymin:ymax,xmin:xmax], net, noiseModel, device)
            means[ymin:ymax,xmin:xmax][ovTop:,ovLeft:] = a[ovTop:,ovLeft:]
            mseEst[ymin:ymax,xmin:xmax][ovTop:,ovLeft:] = b[ovTop:,ovLeft:]
            ymin=ymin-overlap+ps
            ymax=ymin+ps
            ovTop=overlap//2
        ymin=0
        ymax=ps
        xmin=xmin-overlap+ps
        xmax=xmin+ps
        ovLeft=overlap//2
    return means, mseEst