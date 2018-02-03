
<p align="center">
<img src="./etc/img/logo.png" width="800">
</p>

**Tensorbag** is a collection of tensorflow tutorial on different Deep Learning and Machine Learning algorithms. The tutorials are organised as **jupyter notebooks** and require *tensorflow >= 1.5*. There is a subset of notebooks identified with the tag **quiz** that directly ask to the reader to complete part of the code. In the same folder there is always a complementary notebook with the complete solution.

1. **MNIST**: it is a famous dataset of handwritten digits that is commonly used as benchmark in Deep Learning. The tutorial shows how to download and prepare it for training. [[notebook]](./mnist/mnist.ipynb)

2. **CIFAR-10**: it is a dataset of 50k 32x32 color training images, labeled over 10 categories, and 10k test images. The tutorial shows where to download and how to prepare the CIFAR-10 dataset. The pre-processing is done in different ways using Numpy, Tensorflow datasets, and TFRecords. [[notebook]](./cifar10/cifar10.ipynb)

3. **Generative Adversarial Network (GAN)**: implementation of a standard GAN as in the [original paper](https://arxiv.org/pdf/1406.2661.pdf) of Goodfellow et al. (2014). The GAN is trained and tested on the CIFAR-10 dataset. [[notebook]](./generative_adversarial_networks/generative_adversarial_networks.ipynb)
