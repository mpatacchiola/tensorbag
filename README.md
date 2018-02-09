
<p align="center">
<img src="./etc/img/logo.png" width="800">
</p>

**Tensorbag** is a collection of tensorflow tutorial on different Deep Learning and Machine Learning algorithms. The tutorials are organised as **jupyter notebooks** and require *tensorflow >= 1.5*. There is a subset of notebooks identified with the tag **quiz** that directly ask to the reader to complete part of the code. In the same folder there is always a complementary notebook with the complete solution.

1. **MNIST**: it is a famous dataset of handwritten digits that is commonly used as benchmark in Deep Learning. It has 60k training images, and 10k test images. The tutorial shows how to download and prepare the dataset. The pre-processing produces the training and test files in TFRecord format. [[notebook]](./mnist/mnist.ipynb)

2. **CIFAR-10**: it is a dataset of 50k 32x32 color training images, labeled over 10 categories, and 10k test images. The tutorial shows where to download and how to prepare the CIFAR-10 dataset. The pre-processing is done in different ways (Numpy and Tensorflow datasets) and produces training and test files in TFRecord format. [[notebook]](./cifar10/cifar10.ipynb)

3. **LeNet-5 Convolutional Neural Network**: implementation of a standard LeNet-5 CNN as described in the [paper](http://www.dengfanxin.cn/wp-content/uploads/2016/03/1998Lecun.pdf) of LeCun et al. (1998). LeNet-5 was designed for handwritten and machine-printed character recognition. In this tutorial the network is trained and tested on the MNIST dataset. [[notebook]](./lenet5/lenet5.ipynb)

4. **Generative Adversarial Network (GAN)**: implementation of a standard GAN as in the [original paper](https://arxiv.org/pdf/1406.2661.pdf) of Goodfellow et al. (2014). The GAN is trained and tested on the CIFAR-10 dataset. [[notebook]](./generative_adversarial_networks/generative_adversarial_networks.ipynb)
