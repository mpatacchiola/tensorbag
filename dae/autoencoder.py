#!/usr/bin/env python

#The MIT License (MIT)
#Copyright (c) 2018 Massimiliano Patacchiola
#
#THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF 
#MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY 
#CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE 
#SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#Fully connected Autoencoder
#The class is an implementation of an Autoencoder that can be used
#to train a model on the MNIST dataset. The class is flexible enough and
#can be readapted to other datasets. Methods such as save() and load()
#allow the user to save and restore the network. A log file is locally stored
#and can be used to visualize the training from tensorboard.

import tensorflow as tf
import numpy as np
import datetime
import os

class Autoencoder:
    def __init__(self, sess, input_shape, hidden_size_large, hidden_size_medium, hidden_size_small, code_size, output_size, gradient_clip=False, dir_header="./"):
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
        #Encoding-Decoding on the first-style image
        with tf.variable_scope("Encoder", reuse=False):
            self.x = tf.placeholder(tf.float32, input_shape)
            self.x_reshaped = tf.reshape(self.x, shape=[-1, 28, 28, 1])
            self.h1_encoder = tf.layers.dense(self.x, hidden_size_large, activation=activation_function, kernel_initializer=weight_initializer, name="dense_1")
            self.h2_encoder = tf.layers.dense(self.h1_encoder, hidden_size_medium, activation=activation_function, kernel_initializer=weight_initializer, name="dense_2")
            self.h3_encoder = tf.layers.dense(self.h2_encoder, hidden_size_small, activation=activation_function, kernel_initializer=weight_initializer, name="dense_3")
            self.code = tf.layers.dense(self.h3_encoder, code_size, activation=code_activation_function, kernel_initializer=weight_initializer, name="dense_4")           
        with tf.variable_scope("Decoder", reuse=False):
            self.h1_decoder = tf.layers.dense(self.code, hidden_size_small, activation=activation_function, kernel_initializer=weight_initializer, name="dense_1")
            self.h2_decoder = tf.layers.dense(self.h1_decoder, hidden_size_medium, activation=activation_function, kernel_initializer=weight_initializer, name="dense_2")
            self.h3_decoder = tf.layers.dense(self.h2_decoder, hidden_size_large, activation=activation_function, kernel_initializer=weight_initializer, name="dense_3")
            self.output = tf.layers.dense(self.h3_decoder, output_size, activation=output_activation_function, kernel_initializer=weight_initializer, name="dense_4")
            self.output_reshaped = tf.reshape(self.output, shape=[-1, 28, 28, 1])
        #Train operations
        with tf.variable_scope("Training"):
            self.loss = tf.reduce_mean(tf.pow(self.x - self.output, 2))
            if(gradient_clip == False):
                self.train_op = tf.train.AdamOptimizer(learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-08).minimize(self.loss)
            else:
                optimizer = tf.train.AdamOptimizer(learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-08)
                gradients = optimizer.compute_gradients(self.loss)
                capped_gradients = [(tf.clip_by_value(grad, -1., 1.), var) for grad, var in gradients]
                self.train_op = optimizer.apply_gradients(capped_gradients)
            self.tf_saver = tf.train.Saver()
            self.train_iteration = 0
        #Summaries
        with tf.variable_scope("Summaries"):
            tf.summary.image("input_images", self.x_reshaped, max_outputs=6, family="original")
            tf.summary.image("reconstruction_images", self.output_reshaped, max_outputs=6, family="reconstructed")
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


def main():
    batch_size = 128
    tot_epochs = 10
    tot_iterations = (60000 / batch_size) * tot_epochs
    save_every_iteration = tot_iterations - 1
    print_every_iteration = 100

    #Declare session and autoencoder
    sess = tf.Session()
    my_autoencoder = Autoencoder(sess=sess,
                                 input_shape=[None,28*28*1],
                                 hidden_size_large = 256,
                                 hidden_size_medium = 128,
                                 hidden_size_small = 64,
                                 code_size = 32,
                                 output_size = 28*28*1,
                                 gradient_clip=True,
                                 dir_header="./")
    sess.run(tf.global_variables_initializer()) #WARNING: do not call it, if load() method is used

    #This is a Tensorflow automated procedure to download the MNIST dataset
    print("Downloading the MNIST dataset...")
    from tensorflow.examples.tutorials.mnist import input_data
    mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

    for iteration in range(tot_iterations):
        input_feature = mnist.train.next_batch(batch_size)[0]
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
