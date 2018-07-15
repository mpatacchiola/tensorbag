#!/usr/bin/env python

#The MIT License (MIT)
#Copyright (c) 2018 Massimiliano Patacchiola
#
#THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF 
#MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY 
#CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE 
#SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#Convolutional Autoencoder. 
#The decoder is obtained via upsampling and convolution.
#The class is an implementation of an Autoencoder that can be used
#to train a model on the SVHN dataset. The class is flexible enough and
#can be readapted to other datasets. Methods such as save() and load()
#allow the user to save and restore the network. A log file is locally stored
#and can be used to visualize the training from tensorboard.

import tensorflow as tf
import numpy as np
import datetime
import os
import scipy.io as sio

class Autoencoder:
    def __init__(self, sess, conv_filters_large, conv_filters_medium, code_size, dir_header="./"):
        '''Init method
        @param sess (tf.Session) the current session
        @param input_shape (list) the shape of the input layer
        @param hidden_size_* (int) the number of units in the hidden layers
        @param code_size (int) the number of units in the code layer
        @param output_size (int) tipically same as input_shape
        @param gradient_clip (bool) applies gradient clipping on the gradient vector
        '''
        self.dir_header = dir_header
        activation_function = tf.nn.leaky_relu
        code_activation_function = tf.nn.leaky_relu 
        output_activation_function = tf.nn.sigmoid
        weight_initializer = None
        #Encoding-Decoding on the image
        with tf.variable_scope("Encoder", reuse=False):
            #Input
            self.x = tf.placeholder(tf.float32, [None, 32, 32, 3])
            #self.x = tf.reshape(self.x, [-1, 32, 32, 3])
            #Encoder
            self.conv_1_1 = tf.layers.conv2d(inputs=self.x, filters=conv_filters_large, kernel_size=[5, 5], padding="same", activation=tf.nn.relu)
            self.pool_1_1 = tf.layers.max_pooling2d(inputs=self.conv_1_1, pool_size=[2, 2], strides=2)
            self.conv_1_2 = tf.layers.conv2d(inputs=self.pool_1_1, filters=conv_filters_medium, kernel_size=[5, 5], padding="same", activation=tf.nn.relu)
            self.pool_1_2 = tf.layers.max_pooling2d(inputs=self.conv_1_2, pool_size=[2, 2], strides=2)
            self.pool_1_2_flat = tf.reshape(self.pool_1_2, [-1, 8 * 8 * conv_filters_medium])
            #Code
            self.code = tf.layers.dense(inputs=self.pool_1_2_flat, units=code_size, activation=tf.nn.relu)
            #Decoder
            self.dense_2_2 = tf.layers.dense(self.code, units=8 * 8 * conv_filters_medium, activation=tf.nn.relu)
            self.dense_2_2_reshaped = tf.reshape(self.dense_2_2, [-1, 8, 8, conv_filters_medium])
            self.upsample_2_1 = tf.image.resize_nearest_neighbor(self.dense_2_2_reshaped, size=[16, 16])
            print("upsample_2_1: " + str(self.upsample_2_1))
            #self.deconv1 = tf.layers.conv2d_transpose(inputs=self.code, filters=32, kernel_size[5,5], strides=(1, 1), padding='same', activation=tf.nn.relu)
            self.conv_2_1 = tf.layers.conv2d(inputs=self.upsample_2_1, filters=conv_filters_medium, kernel_size=[7, 7], padding="same", activation=tf.nn.relu)
            print("conv_2_1: " + str(self.conv_2_1))
            self.upsample_2_2 = tf.image.resize_nearest_neighbor(self.conv_2_1, size=[32, 32])
            print("upsample_2_2: " + str(self.upsample_2_2))
            self.conv_2_2 = tf.layers.conv2d(inputs=self.upsample_2_2, filters=conv_filters_large, kernel_size=[7, 7], padding="same", activation=tf.nn.relu)
            print("conv_2_2: " + str(self.conv_2_2))
            self.output = tf.layers.conv2d(inputs=self.conv_2_2, filters=3, kernel_size=[7, 7], padding="same", activation=tf.nn.sigmoid)
            print("output: " + str(self.output))

        #Train operations
        with tf.variable_scope("Training"):
            self.loss = tf.reduce_mean(tf.pow(tf.subtract(self.x,self.output), 2))
            #self.loss = tf.reduce_mean(tf.pow(tf.subtract(self.pool_1_2, self.code_reshaped), 2))
            self.train_op = tf.train.AdamOptimizer(learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-08).minimize(self.loss)
            self.tf_saver = tf.train.Saver()
            self.train_iteration = 0
        #Summaries
        with tf.variable_scope("Summaries"):
            tf.summary.image("input_images", self.x, max_outputs=5, family="original")
            tf.summary.image("reconstruction_images", self.output, max_outputs=5, family="reconstructed")
            tf.summary.scalar("loss_reconstruction", self.loss, family="losses")
            tf.summary.histogram("hist_code", self.code, family="code")
            summary_folder = self.dir_header + '/log/' + str(datetime.datetime.now().time())
            self.tf_summary_writer = tf.summary.FileWriter(summary_folder, sess.graph)
            self.summaries = tf.summary.merge_all() #merge all the previous summaries

    def forward(self, sess, input_feature):
        '''Feed-forward pass in the autoencoder
        @param sess (tf.Session) the current session
        @param input_feature (np.array) matrix or array of feature
        @return (np.array) the output of the autoencoder
        '''
        output = sess.run([self.output], feed_dict={self.x: input_feature})
        return output

    def test(self, sess, input_feature):
        '''Single step test of the autoencoder
        @param sess (tf.Session) the current session
        @param input_feature (np.array) matrix or array of feature
        @return (float) the loss
        '''
        loss = sess.run([self.loss], feed_dict={self.x: input_feature})
        return loss   

    def train(self, sess, input_feature, summary_rate=250):
        '''Single step training of the autoencoder
        @param sess (tf.Session) the current session
        @param input_feature (np.array) matrix or array of feature
        @return (float) the loss
        '''
        _, loss, code, summ = sess.run([self.train_op, self.loss, self.code, self.summaries], feed_dict={self.x: input_feature})
        if(self.train_iteration % summary_rate == 0):
            self.tf_summary_writer.add_summary(summ, global_step=self.train_iteration)
            self.tf_summary_writer.flush()
        self.train_iteration += 1
        return loss, code

    def save(self, sess, verbose=True):
        '''Save the model
        @param sess (tf.Session) the current session
        @param verbose (bool) if True print information on terminal
        '''
        if not os.path.exists(self.dir_header + "/model/"):
            os.makedirs(self.dir_header + "/model/")
        model_folder = self.dir_header + "/model/" + str(datetime.datetime.now().time()) + "_" + str(self.train_iteration) + "/model.ckpt"
        if(verbose): print("Saving networks in: " + str(model_folder))    
        save_path = self.tf_saver.save(sess, model_folder)

    def load(self, sess, file_path, verbose=True):
        '''Load a model
        NOTE: when loading a model the method tf.global_variables_initializer()
        must not be called otherwise the variables are set to random values
        @param sess (tf.Session) the current session
        @param verbose (bool) if True print information on terminal
        '''
        if(verbose): print("Loading networks from: " + str(file_path))
        save_path = self.tf_saver.restore(sess, file_path)
        if(verbose): print("Done!")

def download_dataset(save_path, verbose=True):
    import urllib
    if(os.path.isfile(save_path + "train_32x32.mat") == False):
        if(verbose): print("Downloading train_32x32.mat...")
        urllib.urlretrieve ("http://ufldl.stanford.edu/housenumbers/train_32x32.mat", save_path + "train_32x32.mat")
        if(verbose): print("Done!")
    if(os.path.isfile(save_path + "test_32x32.mat") == False):
        if(verbose): print("Downloading test_32x32.mat...")
        urllib.urlretrieve ("http://ufldl.stanford.edu/housenumbers/test_32x32.mat", save_path + "test_32x32.mat")
        if(verbose): print("Done!")

def mat_to_numpy(mat_path, tot_features, verbose=True):
    train_mat = sio.loadmat(mat_path)
    train_images_array = train_mat["X"]
    train_images_array = np.swapaxes(train_images_array, 2, 3) #[32, 32, 3, None] > [32, 32, None, 3]
    train_images_array = np.swapaxes(train_images_array, 1, 2) #[32, 32, None, 3] > [32, None, 32, 3]
    train_images_array = np.swapaxes(train_images_array, 0, 1) #[32, None, 32, 3] > [None, 32, 32, 3]
    train_images_array = np.reshape(train_images_array, [tot_features, 32*32*3])
    if(verbose): print(type(train_images_array))
    if(verbose): print(train_images_array.shape)
    train_labels_array = train_mat["y"]
    if(verbose): print(type(train_labels_array))
    if(verbose): print(train_labels_array.shape)
    return train_images_array, train_labels_array

def main():

    batch_size = 64
    tot_epochs = 10
    tot_iterations = (73257 / batch_size) * tot_epochs
    save_every_iteration = tot_iterations - 1
    print_every_iteration = 100

    #Declare session and autoencoder
    sess = tf.Session()
    my_autoencoder = Autoencoder(sess=sess,
                                 conv_filters_large = 32,
                                 conv_filters_medium = 16,
                                 code_size = 64,
                                 dir_header="./")
    sess.run(tf.global_variables_initializer()) #WARNING: do not call it, if load() method is used

    #This is a Tensorflow automated procedure to download the MNIST dataset
    print("Downloading the SVHN dataset...")
    download_dataset(save_path="./")
    train_images_array, train_labels_array = mat_to_numpy(mat_path="./train_32x32.mat", tot_features=73257)
    train_images_array = train_images_array/ 255.0
    print("Training shape: " + str(train_images_array))

    for iteration in range(tot_iterations):
        random_indices = np.random.randint(0, 73257, batch_size)
        input_feature = np.take(train_images_array, random_indices, axis=0)
        input_feature = np.reshape(input_feature, [batch_size, 32, 32, 3])
        local_loss, local_code = my_autoencoder.train(sess, input_feature, summary_rate=25)
        if(iteration % print_every_iteration == 0):
            print("Iteration: " + str(iteration))
            print("Loss: " + str(local_loss))
            print("Code sample: " + str(local_code[0,:]))
            print("")
        if(iteration % save_every_iteration == 0):
            my_autoencoder.save(sess, verbose=True)

if __name__ == "__main__":
    main()
