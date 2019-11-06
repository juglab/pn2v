#!/usr/bin/env python3

import os
import sys
import argparse

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--histogram", help="name of .npy-file containing the noise model histogram", default='noiseModel.npy')
parser.add_argument("--name", help="name of your network", default='N2V')
parser.add_argument("--dataPath", help="The path to your training data")
parser.add_argument("--fileName", help="name of your training data file", default="*.tif")
parser.add_argument("--validationFraction", help="Fraction of data you want to use for validation (percent)", default=5.0, type=float)
#parser.add_argument("--dims", help="dimensions of your data, can include: X,Y,Z,C (channel), T (time)", default='YX')
parser.add_argument("--patchSizeXY", help="XY-size of your training patches", default=64, type=int)
#parser.add_argument("--patchSizeZ", help="Z-size of your training patches", default=64, type=int)
parser.add_argument("--epochs", help="number of training epochs", default=200, type=int)
parser.add_argument("--stepsPerEpoch", help="number training steps per epoch", default=50, type=int)
parser.add_argument("--batchSize", help="size of your training batches", default=4, type=int)
parser.add_argument("--virtualBatchSize", help="size of virtual batch", default=20, type=int)
parser.add_argument("--netDepth", help="depth of your U-Net", default=2, type=int)


parser.add_argument("--learningRate", help="initial learning rate", default=1e-3, type=float)

parser.add_argument("--netKernelSize", help="Size of conv. kernels in first layer", default=3, type=int)
parser.add_argument("--n2vPercPix", help="percentage of pixels to manipulated by N2V", default=1.6, type=float)
parser.add_argument("--unet_n_first", help="number of feature channels in the first u-net layer", default=32, type=int)

if len(sys.argv)==1:
    parser.print_help(sys.stderr)
    sys.exit(1)

args = parser.parse_args()
print(args)

import matplotlib.pyplot as plt
import numpy as np
from unet.model import UNet

from pn2v import utils
from pn2v import histNoiseModel
from pn2v import training
from tifffile import imread
import glob
# See if we can use a GPU
device=utils.getDevice()


import glob

print("args",str(args.name))


####################################################
#           PREPARE TRAINING DATA
####################################################

path=args.dataPath
files=glob.glob(path+args.fileName)
# Load the training data
data=[]
for f in files:
    data.append(imread(f).astype(np.float32))
    print('loading',f)

data =np.array(data)
print(data.shape)

if len(data.shape)==4:
    data.shape=(data.shape[0]*data.shape[1],data.shape[2],data.shape[3])

print(data.shape)




####################################################
#           PREPARE Noise Model
####################################################

histogram=np.load(path+args.histogram)

# Create a NoiseModel object from the histogram.
noiseModel=histNoiseModel.NoiseModel(histogram, device=device)


####################################################
#           CREATE AND TRAIN NETWORK
####################################################

net = UNet(800, depth=args.netDepth)

# Split training and validation data.
my_train_data=data[:-5].copy()
np.random.shuffle(my_train_data)
my_val_data=data[-5:].copy()
np.random.shuffle(my_val_data)

# Start training.
trainHist, valHist = training.trainNetwork(net=net, trainData=my_train_data, valData=my_val_data,
                                           postfix=args.name, directory=path, noiseModel=noiseModel,
                                           device=device, numOfEpochs= args.epochs, stepsPerEpoch=args.stepsPerEpoch,
                                           virtualBatchSize=args.virtualBatchSize, batchSize=args.batchSize,
                                           learningRate=args.learningRate,
                                           augment=False)

