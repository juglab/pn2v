# Probabilistic Noise2Void

This repository hosts our implementation of Probabilistic Noise2Void. The original paper can be found at https://arxiv.org/abs/1906.00651. PN2V is self-supervised CNN-based-denoising that achieves results close to state-of-the-art methods, but requires only individual noisy images for training.
Requirements:
* Pytorch: https://pytorch.org/get-started/locally/

# Examples:

Checkout our example notebooks. Please use them in the order stated below:

* Creating a noise model: Convallaria-1-CreateNoiseModel.ipynb
* Training a network: Convallaria-2-Training.ipynb
* Predictig, i.e. denoising images: Convallaria-3-Prediction.ipynb
