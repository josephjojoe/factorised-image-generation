# factorised-image-generation
A collection of GANs demonstrating the efficiency of factorised transposed convolution (deconvolution) layers - performance is comparable to conventional model architectures, but with ~15-40% less parameters.


## Context
The Inception architecture [1] and variants such as Xception [2] have been used in image classification tasks with great success - successive versions Inception v2 and v3 have improved on the naive implementation by factorising larger convolutions such as 5x5 to two 3x3 convolution operations as well as by factorising operations of the form NxN to two operations 1xN and Nx1, reducing the amount of needed parameters whilst maintaining a similarly high ability to recognise patterns in input images.

In this repository, this concept of factorising convolutional layers is applied in reverse to transposed convolution layers; a copy of the model without factorised layers is also present within each folder for easy comparison of the two. The factorised models contain around 15-40% less parameters, but perform approximately as well (subjectively) as their naive counterparts.


## Examples

### CIFAR-10 (Frog)

Factorised model             |  Naive model
:-------------------------:|:-------------------------:
<img src="examples/cifar-10/factorised_generated_frog9800.png" width="150" height="150"/> <img src="examples/cifar-10/factorised_generated_frog9900.png" width="150" height="150"/> <br> <img src="examples/cifar-10/factorised_generated_frog4900.png" width="150" height="150"/> <img src="examples/cifar-10/factorised_generated_frog5100.png" width="150" height="150"/> | <img src="examples/cifar-10/naive_generated_frog9800.png" width="150" height="150"/> <img src="examples/cifar-10/naive_generated_frog9900.png" width="150" height="150"/> <br> <img src="examples/cifar-10/naive_generated_frog4900.png" width="150" height="150"/> <img src="examples/cifar-10/naive_generated_frog5100.png" width="150" height="150"/>

### MNIST

Factorised model             |  Naive model
:-------------------------:|:-------------------------:
<img src="examples/mnist/factorised_mnist_numbers.gif" width="250" height="250"/> | <img src="examples/mnist/naive_mnist_numbers.gif" width="250" height="250"/>

### EMNIST Letters

Factorised model             |  Naive model
:-------------------------:|:-------------------------:
<img src="examples/emnist/factorised_emnist_letters.gif" width="250" height="250"/> | <img src="examples/emnist/naive_emnist_letters.gif" width="250" height="250"/>


## Improved Parameter Efficiency
The table below lists the reduction in trainable parameters gained by factorising transposed convolution layers in the generator component of the GAN for each dataset. Please note that the parameter numbers shown here refer to **all** parameters, trainable and non-trainable, and that they only include the generator of each GAN and not the discriminator - I have not factored the convolutional discriminator layers, hence the parameters in both the naive and factorised implementations would be the same.

Dataset                    |  Naive generator model parameters | Factorised generator model parameters | Parameter reduction
:-------------------------:|:---------------------------------:|:-------------------------------------:|:-------------------------:
MNIST                      | 2,330,944                         | 1,558,293                             | 772,651 (31.15%)
CIFAR-10                   | 6,264,579                         | 3,971,587                             | 2,292,992 (36.60%)
EMNIST Letters             | 4,287,808                         | 3,570,245                             | 717,563 (16.73%)


## Code
The [MNIST model code](https://www.tensorflow.org/tutorials/generative/dcgan) is from the TensorFlow Deep Convolutional Generative Adversarial Network guide.

The [CIFAR-10 model code](https://colab.research.google.com/github/fchollet/deep-learning-with-python-notebooks/blob/master/chapter12_part05_gans.ipynb) is from Deep Learning with Python, by Francois Chollet.

Other model code has also generally been based upon the above resources with some modifications for data loading and pre-processing.

Visible flickering was initially present in the GIFs for the MNIST and EMNIST Letters datasets, indicating a certain amount of model instability - this was rectified by reducing the the momentum term β1 from 0.9 to 0.5, as per the work of Radford et al. [3], and increasing the learning rate to 0.0002 to compensate.


## Datasets
The datasets used in this repository are:
- MNIST database of handwritten digits
- CIFAR-10
- EMNIST Letters

## Further Steps
I plan to use the bilinear additive upsampling method described by Wojna et al. [4] and the idea of layer branching from the Inception module architecture to produce a novel GAN architecture, entitled InceptionGAN.

## Citations 

```
[1] Szegedy, C. et al., 2014. Going deeper with convolutions. arXiv.

[2] Chollet, F., 2017. Xception: Deep Learning with Depthwise Separable Convolutions. arXiv.

[3] Radford, A., Metz, L. & Chintala, S., 2016. Unsupervised Representation Learning 
with Deep Convolutional Generative Adversarial Networks. arXiv.

[4] Wojna, Z. et al., 2019. The Devil is in the Decoder: Classification, Regression and GANs. arXiv.
```
