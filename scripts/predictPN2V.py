import os
import sys
import argparse
import glob

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--baseDir", help="directory in which all your network will live", default='models')
parser.add_argument("--name", help="name of your network", default='N2V')
parser.add_argument("--dataPath", help="The path to your data")
parser.add_argument("--fileName", help="name of your data file", default="*.tif")
parser.add_argument("--output", help="The path to which your data is to be saved", default='.')
parser.add_argument("--tileSize", help="width/height of tiles used to make it fit GPU memory", default=256, type=int)
parser.add_argument("--tileOvp", help="overlap of tiles used to make it fit GPU memory", default=48, type=int)
parser.add_argument("--histogram", help="name of .npy-file containing the noise model histogram", default='noiseModel.npy')

if len(sys.argv)==1:
    parser.print_help(sys.stderr)
    sys.exit(1)

args = parser.parse_args()

print(args.output)

# We import all our dependencies.
import numpy as np
from matplotlib import pyplot as plt
from tifffile import imread
from tifffile import imwrite

import torch
from unet.model import UNet
from pn2v.utils import denormalize
from pn2v.utils import normalize
from pn2v import utils
from pn2v import prediction
import pn2v.training
from pn2v import histNoiseModel


device=utils.getDevice()
path=args.dataPath


####################################################
#           PREPARE Noise Model
####################################################

histogram=np.load(path+args.histogram)
noiseModel=histNoiseModel.NoiseModel(histogram, device=device)



####################################################
#           LOAD NETWORK
####################################################

net=torch.load(path+"/last_"+args.name+".net")


####################################################
#           PROCESS DATA
####################################################


files=glob.glob(path+args.fileName)

for f in files:
    print('loading',f)
    image= (imread(f).astype(np.float32))
    if len(image.shape)<3:
        image=image[np.newaxis,...]


    means=np.zeros(image.shape)
    mseEst=np.zeros(image.shape)
    for i in range (image.shape[0]):
        im=image[i,...]
        # processing image
        means[i,...], mseEst[i,...] = prediction.tiledPredict(im, net ,ps=args.tileSize, overlap=args.tileOvp,
                                            device=device,
                                            noiseModel=noiseModel)

    if im.shape[0]==1:
        means=means[0]
        mseEst=mseEst[0]

    outpath=args.output
    filename=os.path.basename(f).replace('.tif','_MMSE-PN2V.tif')
    outpath=os.path.join(outpath,filename)
    print('writing',outpath)
    imwrite(outpath, mseEst)

    outpath=args.output
    filename=os.path.basename(f).replace('.tif','_Prior-PN2V.tif')
    outpath=os.path.join(outpath,filename)
    print('writing',outpath)
    imwrite(outpath, means)




