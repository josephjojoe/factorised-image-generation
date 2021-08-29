# factorised-image-generation
GANs demonstrating the efficiency of factorised transposed convolution layers - comparable to conventional models, but with ~25-35% less parameters.

## Context
The Inception architecture and variants such as Xception have been used in image classification tasks with great success - instead of stacking convolutional layers, Inception modules contain filters of varying sizes that are applied concurrently to the input data. Successive model versions Inception v2 and v3 have improved on the naive implementation by factorising larger convolutions such as 5x5 to two 3x3 convolution operations and in turn factorising operations of the form NxN to two successive operations 1xN and Nx1, reducing the amount of needed parameters whilst maintaining a similarly high ability to recognise patterns in input images.
