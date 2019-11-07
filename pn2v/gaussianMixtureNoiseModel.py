import torch
dtype = torch.float
device = torch.device("cuda:0") 
import matplotlib.pyplot as plt
import numpy as np
import pickle
from torch.distributions import normal
from scipy.stats import norm
from tifffile import imread

import pn2v.utils as utils
import pn2v.histNoiseModel

class GaussianMixtureNoiseModel:
    
    
    def __init__(self, min_signal, max_signal, weight, n_gaussian=1, n_coeff=2, noiseModelTraining=True):
        if(weight is None):
            weight = np.random.randn(n_gaussian*3, n_coeff)
            weight[n_gaussian:2*n_gaussian, 1]=np.log(max_signal-min_signal) # exp(b) = max_observation for Poisson
            weight = torch.from_numpy(weight.astype(np.float32)).float().to(device)
            weight.requires_grad=True 
        self.n_gaussian = weight.shape[0]//3
        self.weight = weight
        self.n_coeff = n_coeff
        self.min_signal=torch.Tensor([min_signal]).to(device)
        self.max_signal=torch.Tensor([max_signal]).to(device)
        self.min_signal_scal=min_signal
        self.max_signal_scal=max_signal
        self.tol=torch.Tensor([1e-10]).to(device)
        self.noiseModelTraining=noiseModelTraining
        
    
    def polynomialRegressor(self, params, s):
        value=0
        for i in range(params.shape[0]):
            value +=params[i]*( ( (s-self.min_signal) / (self.max_signal-self.min_signal) ) ** i); # use this for training network   
        return value
        
   
    def normalDens(self, x,m_=0.0,std_=None):
        tmp=-((x-m_)**2) 
        tmp=tmp / (2.0*std_*std_)
        tmp= torch.exp(tmp )
        tmp= tmp/ torch.sqrt( (2.0*np.pi)*std_*std_)
        return tmp

    def likelihood(self, obs, signal):
        gaussianParameters=self.getGaussianParameters(signal, False)
        p=0
        for gaussian in range(self.n_gaussian):
            p+=self.normalDens(obs, gaussianParameters[gaussian], 
                               gaussianParameters[self.n_gaussian+gaussian])*gaussianParameters[2*self.n_gaussian+gaussian]
        return p+self.tol
    
    def getGaussianParameters(self, signal, torchTensor = True):
    
        noiseModel = []
        mu = []
        sigma = []
        alpha = []
        kernels = self.weight.shape[0]//3
        for num in range(kernels):
            mu.append(self.polynomialRegressor(self.weight[num, :], signal))
            
            sigmaTemp=self.polynomialRegressor(torch.exp(self.weight[kernels+num, :]), signal)
            sigmaTemp = torch.clamp(sigmaTemp, min = 50)
            sigma.append(torch.sqrt(sigmaTemp))
#             sigma.append(torch.sqrt(self.polynomialRegressor(self.weight[kernels+num, :], signal-self.min_signal-self.tol)))
            alpha.append(torch.exp(self.polynomialRegressor(self.weight[2*kernels+num, :], signal)+self.tol))

        sum_alpha = 0
        for al in range(kernels):
            sum_alpha = alpha[al]+sum_alpha
        for ker in range(kernels):
            alpha[ker]=alpha[ker]/sum_alpha
            
        sum_means = 0
        for ker in range(kernels):
            sum_means = alpha[ker]*mu[ker]+sum_means

        mu_shifted=[]
        for ker in range(kernels):
            mu[ker] = mu[ker]-sum_means+signal
        
        for i in range(kernels):
            noiseModel.append(mu[i])
        for j in range(kernels):
            noiseModel.append(sigma[j])
        for k in range(kernels):
            noiseModel.append(alpha[k])

        return noiseModel
    

    def train(self, sig_obs_pairs, learning_rate=1e-1, batchSize=1000, n_epochs=1000, name= 'trained_weights_minSignal_maxSignal'):
        counter=0
        optimizer = torch.optim.Adam([self.weight], lr=learning_rate)
        for t in range(n_epochs):

            jointLoss=0
            if (counter+1)*batchSize >= sig_obs_pairs.shape[0]:
                counter=0
                sig_obs_pairs=utils.fastShuffle(sig_obs_pairs,1)

            batch_vectors = sig_obs_pairs[counter*batchSize:(counter+1)*batchSize, :]
            batch = batch_vectors[:,1].astype(np.float32)
            signal = batch_vectors[:,0].astype(np.float32)

            batch = torch.from_numpy(batch.astype(np.float32)).float().to(device)
            signal = torch.from_numpy(signal).float().to(device)

            p = self.likelihood(batch, signal)


            loss=torch.mean(-torch.log(p))
            jointLoss=jointLoss+loss
            
            if t%100==0:
                print(t, jointLoss.item())
                
            if t%(int(n_epochs*0.5))==0:
                trained_weight = self.weight.cpu().detach().numpy()
                min_signal = self.min_signal.cpu().detach().numpy()
                max_signal = self.max_signal.cpu().detach().numpy()
                np.savez(name+".npz", trained_weight=trained_weight, min_signal = min_signal, max_signal = max_signal)

            optimizer.zero_grad()
            jointLoss.backward()
            optimizer.step()
            counter+=1