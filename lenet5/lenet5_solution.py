#!/usr/bin/env python

#The MIT License (MIT)
#Copyright (c) 2018 Massimiliano Patacchiola
#
#THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF 
#MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY 
#CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE 
#SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#Possible solution for the quiz
#Training and testing LeNet5 on CIFAR-10

import numpy as np
import cPickle
import tensorflow as tf
import time
import datetime
import os
#os.environ["CUDA_VISIBLE_DEVICES"] = "0"

class LeNet5:

    def __init__(self, output_size, verbose=True):
        network_id = "lenet5"
        with tf.name_scope("network_" + network_id):
                #Defining variables, biases, and weights
                self.x_placeholder = tf.placeholder(tf.float32, shape=[None, 32, 32, 3])
                self.labels_placeholder = tf.placeholder(tf.int32, shape=[None])
                c1 = tf.layers.conv2d(inputs=self.x_placeholder, filters=6, kernel_size=[5, 5], 
                                      padding="valid", activation=tf.nn.relu)
                s2 = tf.layers.max_pooling2d(inputs=c1, pool_size=[2, 2], strides=2, padding="valid")
                c3 = tf.layers.conv2d(inputs=s2, filters=16, kernel_size=[5, 5], 
                                      padding="valid", activation=tf.nn.relu) 
                s4 = tf.layers.max_pooling2d(inputs=c3, pool_size=[2, 2], strides=2)
                s4_flat = tf.reshape(s4, [-1, 5 * 5 * 16])   
                c5 = tf.layers.dense(inputs=s4_flat, units=120, activation=tf.nn.relu)   
                f6 = tf.layers.dense(inputs=c5, units=84, activation=tf.nn.relu)
                output = tf.layers.dense(inputs=f6, units=output_size, activation=None)
                softmax_output = tf.nn.softmax(output, name="softmax")
                argmax_output = tf.argmax(output, axis=1, name="argmax")
                #Train
                #optimizer= tf.train.MomentumOptimizer(learning_rate=0.1, momentum=0.9, use_locking=False, name='Momentum', use_nesterov=True)
                optimizer= tf.train.AdamOptimizer (learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-08, use_locking=False, name='Adam')
                self.loss = tf.losses.sparse_softmax_cross_entropy(labels=self.labels_placeholder, logits=output)
                accuracy, self.acc_op = tf.metrics.accuracy(self.labels_placeholder, argmax_output)
                self.train = optimizer.minimize(self.loss, name="train")

                if(verbose):
                         print("======== network_" + network_id + "========")
                         print(c1.name)
                         print(s2.name)
                         print(c3.name)
                         print(s4.name)
                         print(c5.name)
                         print(f6.name)
                         print(output.name)
                         print(argmax_output.name)
                         print(self.loss.name)
                         print(self.acc_op)
                         print(self.train.name)
                         print("")
        self.tf_saver = tf.train.Saver()


    def training(self, sess, one_shot_iterator, print_every, checkpoint_list, verbose=True):
        if(verbose): print("Starting training...")
        start_time = time.time()
        summary_folder = './log/' + str(datetime.datetime.now().time())
        tf_summary_writer = tf.summary.FileWriter(summary_folder, sess.graph)
        relative_epochs = 0
        iteration = 0
        next_element = one_shot_iterator.get_next() # sess.run([iterator])
        while(True):
           try:
               input_array, labels =sess.run(next_element)

               tf_list = [self.loss, self.acc_op, self.train]
               tf_dict = {self.x_placeholder: input_array, self.labels_placeholder: labels}
               output = sess.run(tf_list, tf_dict)

               #Statistics about the training
               local_loss = output[0]
               local_accuracy = output[1]

               #Local summaries for tensorboard
               local_summary = tf.Summary()
               local_summary.value.add(simple_value=local_loss, node_name="loss summary", tag="loss")
               local_summary.value.add(simple_value=local_accuracy, node_name="accuracy summary", tag="accuracy")
               tf_summary_writer.add_summary(local_summary, iteration)
               tf_summary_writer.flush()
               iteration += 1

               #Printing on terminal if verbose=True
               if(iteration % print_every == 0 and verbose == True):
                   relative_epochs += 1
                   if(relative_epochs in checkpoint_list):
                       model_folder = "./model/" + str(datetime.datetime.now().time()) + "_" + str(relative_epochs) + "/model.ckpt"
                       if(verbose): print("Saving networks in: " + str(model_folder))    
                       save_path = self.tf_saver.save(sess, model_folder)                   
                   print("==============================")
                   print("Epoch (relative): " + str(relative_epochs))
                   print("Iteration: " + str(iteration))
                   print("Loss: " + str(local_loss))
                   print("Accuracy: " + str(local_accuracy))
                   print("==============================")
                   print("")
           except tf.errors.OutOfRangeError: 
               break

        stop_time = time.time()
        total_time = stop_time - start_time
        if(verbose): print("Time: " + str(total_time) + " seconds")
        model_folder = "./model/" + str(datetime.datetime.now().time()) + "_" + str(iteration) + "/model.ckpt"
        if(verbose): print("Saving networks in: " + str(model_folder))
        save_path = self.tf_saver.save(sess, model_folder)

    def test(self, sess, one_shot_iterator, verbose=True):
        next_element = one_shot_iterator.get_next() # sess.run([iterator])        
        while(True):
           try:
               input_array, labels =sess.run(next_element)
               tf_list = [self.acc_op]
               tf_dict = {self.x_placeholder: input_array, self.labels_placeholder: labels}
               output = sess.run(tf_list, tf_dict)
               local_accuracy = output[0]
           except tf.errors.OutOfRangeError: 
               break
           print("==============================")
           print("Accuracy: " + str(local_accuracy))
           print("==============================")
           print("")
        return local_accuracy

    def load_dataset(self, dataset_path, tot_epochs, batch_size, flip=False, shuffle=True, verbose=True):
        def _parse_function(example_proto):
            features = {"image": tf.FixedLenFeature((), tf.string, default_value=""),
                        "label": tf.FixedLenFeature((), tf.int64, default_value=0)}
            parsed_features = tf.parse_single_example(example_proto, features)
            image_decoded = tf.decode_raw(parsed_features["image"], tf.uint8) #char -> uint8
            image_R = tf.reshape(image_decoded[0:1024], [32, 32])
            image_G = tf.reshape(image_decoded[1024:2048], [32, 32])
            image_B = tf.reshape(image_decoded[2048:4096], [32, 32])
            image_stack = tf.stack([image_R, image_G, image_B], axis=2)
            image_normalized = tf.multiply(tf.cast(image_stack, tf.float32), 1.0/255.0)
            image_shifted = tf.add(image_normalized, -0.5)
            label = parsed_features["label"]
            return image_shifted, label

        def _parse_mirror_function(example_proto):
            features = {"image": tf.FixedLenFeature((), tf.string, default_value=""),
                        "label": tf.FixedLenFeature((), tf.int64, default_value=0)}
            parsed_features = tf.parse_single_example(example_proto, features)
            image_decoded = tf.decode_raw(parsed_features["image"], tf.uint8) #char -> uint8
            image_R = tf.reshape(image_decoded[0:1024], [32, 32])
            image_G = tf.reshape(image_decoded[1024:2048], [32, 32])
            image_B = tf.reshape(image_decoded[2048:4096], [32, 32])
            image_stack = tf.stack([image_R, image_G, image_B], axis=2)
            image_flipped = tf.image.flip_left_right(image_stack)
            image_normalized = tf.multiply(tf.cast(image_flipped, tf.float32), 1.0/255.0)
            image_shifted = tf.add(image_normalized, -0.5)
            label = parsed_features["label"]
            return image_shifted, label

        if(verbose): print "Loading the training datasets..."
        tf_dataset = tf.data.TFRecordDataset(dataset_path)
        if(verbose): print "Parsing the training datasets..."
        tf_dataset = tf_dataset.map(_parse_function)
        if(flip):
            if(verbose): print "Parsing the 'flipped' training datasets..."
            tf_flipped_dataset = tf.data.TFRecordDataset(dataset_path)
            tf_dataset = tf_dataset.concatenate(tf_flipped_dataset.map(_parse_mirror_function))
        if(verbose):  print "Verifying types and shapes..."
        if(verbose):  print(tf_dataset.output_types)
        if(verbose):  print(tf_dataset.output_shapes)

        tf_dataset = tf_dataset.cache() #cache the dataset in memory

        if(shuffle): 
            tf_dataset = tf_dataset.shuffle(100000)
        tf_dataset = tf_dataset.batch(batch_size)
        tf_dataset = tf_dataset.repeat(tot_epochs)
        iterator = tf_dataset.make_one_shot_iterator()
        return iterator


    def save(self, sess, file_path, verbose=True):
        if(verbose): print("Saving networks in: " + str(file_path))
        save_path = self.tf_saver.save(sess, file_path)
        if(verbose): print("Done!")

    def load(self, sess, file_path, verbose=True):
        if(verbose): print("Loading networks from: " + str(file_path))
        save_path = self.tf_saver.restore(sess, file_path)
        if(verbose): print("Done!")



def main():

    TRAINING = True #set to False to enable testing

    if(TRAINING):
        print("Starting Train...")
        tf_sess = tf.Session()
        my_lenet = LeNet5(output_size = 10)
        train_dataset_path = "./cifar10_train.tfrecord"
        train_iterator = my_lenet.load_dataset(dataset_path=train_dataset_path, tot_epochs=500, batch_size=128, flip=True, shuffle=True, verbose=True)
        tf_sess.run(tf.global_variables_initializer())
        tf_sess.run(tf.local_variables_initializer())
        #Training
        my_lenet.training(sess=tf_sess, one_shot_iterator=train_iterator, print_every=781, checkpoint_list=[50, 100, 200, 300, 400, 500], verbose=True)
    else:
        print("Starting Test...")
        tf_sess = tf.Session()
        my_lenet = LeNet5(output_size = 10)
        my_lenet.load(tf_sess, file_path="./model/18:30:16.502012_300/model.ckpt", verbose=True)
        test_dataset_path = "./cifar10_test.tfrecord"
        test_iterator = my_lenet.load_dataset(dataset_path=test_dataset_path, tot_epochs=1, batch_size=10000, flip=False, shuffle=False, verbose=True)
        tf_sess.run(tf.local_variables_initializer())
        #Test
        my_lenet.test(sess=tf_sess, one_shot_iterator=test_iterator, verbose=True)        


if __name__ == "__main__":
    main()


