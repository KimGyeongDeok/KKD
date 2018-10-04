# encoding: UTF-8
# Copyright 2016 Google.com
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import tensorflow as tf
import MAKE_EXAMPLE_GEN as makex
import numpy as np
import math 
import matplotlib.pyplot as plt

calculate=makex.make_examples_from_binvox('train')
DIMS=calculate[2]
number_of_files=calculate[3]
number_of_class=calculate[4]
batchsize_per_step=13
step_per_1epoch=number_of_files/batchsize_per_step


print("Tensorflow version " + tf.__version__)




tf.set_random_seed(0)

# neural network structure for this sample in example of conv2d
#
# · · · · · · · · · ·      (input data, 1-deep)                 X [batch, -, -, 1]
# @ @ @ @ @ @ @ @ @ @   -- conv. layer 6x6x1=>6 stride 1        W1 [6, 6, 1, 6]        B1 [6]
# ∶∶∶∶∶∶∶∶∶∶∶∶∶∶∶∶∶∶∶                                           Y1 [batch, 28, 28, 6]
#   @ @ @ @ @ @ @ @     -- conv. layer 5x5x6=>12 stride 2       W2 [5, 5, 6, 12]        B2 [12]
#   ∶∶∶∶∶∶∶∶∶∶∶∶∶∶∶                                             Y2 [batch, 14, 14, 12]
#     @ @ @ @ @ @       -- conv. layer 4x4x12=>24 stride 2      W3 [4, 4, 12, 24]       B3 [24]
#     ∶∶∶∶∶∶∶∶∶∶∶                                               Y3 [batch, 7, 7, 24] => reshaped to YY [batch, 7*7*24]
#      \x/x\x\x/ ✞      -- fully connected layer (relu+dropout) W4 [-, 200]       B4 [200]
#       · · · ·                                                 Y4 [batch, 200]
#       \x/x\x/         -- fully connected layer (softmax)      W5 [200, 10]           B5 [10]
#        · · ·                                                  Y [batch, 10]

# input X
X = tf.placeholder(tf.float32, [None,DIMS[0],DIMS[1],DIMS[2], 1])
# label Y_
Y_ = tf.placeholder(tf.float32, [None, number_of_class])
# variable learning rate
lr = tf.placeholder(tf.float32)
# Probability of keeping a node during dropout = 1.0 at test time (no dropout) and 0.75 at training time
pkeep = tf.placeholder(tf.float32)
# step for variable learning rate
step = tf.placeholder(tf.int32)

# three convolutional layers with their channel counts, and a
# fully connected layer (the last layer has NUM_CLASS softmax neurons)
K = 6  # first convolutional layer output depth
L = 12  # second convolutional layer output depth
M = 24  # third convolutional layer
N = 200  # fully connected layer
stride1 = 1 # 1st kernel's stride
stride2 = 2 # 2nd kernel's stride
stride3 = 2 # 3th kernel's stride

# fnn's input size calculate
FNN_INPUT_X=int((DIMS[0]-1)/stride1 + 1)
FNN_INPUT_X=int((FNN_INPUT_X-1)/stride2 + 1)
FNN_INPUT_X=int((FNN_INPUT_X-1)/stride3 + 1)
FNN_INPUT_Y=int((DIMS[1]-1)/stride1 + 1)
FNN_INPUT_Y=int((FNN_INPUT_Y-1)/stride2 + 1)
FNN_INPUT_Y=int((FNN_INPUT_Y-1)/stride3 + 1)
FNN_INPUT_Z=int((DIMS[2]-1)/stride1 + 1)
FNN_INPUT_Z=int((FNN_INPUT_Z-1)/stride2 + 1)
FNN_INPUT_Z=int((FNN_INPUT_Z-1)/stride3 + 1)

#Convolution layers' parameters(weights)
W1 = tf.Variable(tf.truncated_normal([6,6,6, 1, K], stddev=0.1))  # 6x6 patch, 1 input channel, K output channels
B1 = tf.Variable(tf.constant(0.1, tf.float32, [K]))
W2 = tf.Variable(tf.truncated_normal([5,5,5, K, L], stddev=0.1))
B2 = tf.Variable(tf.constant(0.1, tf.float32, [L]))
W3 = tf.Variable(tf.truncated_normal([4,4,4, L, M], stddev=0.1))
B3 = tf.Variable(tf.constant(0.1, tf.float32, [M]))

#FNN layers' parameters(weights)
W4 = tf.Variable(tf.truncated_normal([FNN_INPUT_X * FNN_INPUT_Y*FNN_INPUT_Z*M, N], stddev=0.1))
B4 = tf.Variable(tf.constant(0.1, tf.float32, [N]))
W5 = tf.Variable(tf.truncated_normal([N, number_of_class], stddev=0.1))
B5 = tf.Variable(tf.constant(0.1, tf.float32, [number_of_class]))


# The model
Y1 = tf.nn.relu(tf.nn.conv3d(X, W1, strides=[1, stride1, stride1,stride1, 1], padding='SAME') + B1)
Y2 = tf.nn.relu(tf.nn.conv3d(Y1, W2, strides=[1, stride2, stride2,stride2, 1], padding='SAME') + B2)
Y3 = tf.nn.relu(tf.nn.conv3d(Y2, W3, strides=[1, stride3, stride3,stride3, 1], padding='SAME') + B3)

# reshape the output from the third convolution for the fully connected layer
YY = tf.reshape(Y3, shape=[-1, FNN_INPUT_X * FNN_INPUT_Y*FNN_INPUT_Z*M])

Y4 = tf.nn.relu(tf.matmul(YY, W4) + B4)
YY4 = tf.nn.dropout(Y4, pkeep)
Ylogits = tf.matmul(YY4, W5) + B5
Y = tf.nn.softmax(Ylogits)

# cross-entropy loss function (= -sum(Y_i * log(Yi)) ), normalised for batches of 100  images
# TensorFlow provides the softmax_cross_entropy_with_logits function to avoid numerical stability
# problems with log(0) which is NaN
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=Ylogits, labels=Y_)
cross_entropy = tf.reduce_mean(cross_entropy)*100

# accuracy of the trained model, between 0 (worst) and 1 (best)
correct_prediction = tf.equal(tf.argmax(Y, 1), tf.argmax(Y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


# training step, the learning rate is a placeholder
# the learning rate is: # 0.0001 + 0.003 * (1/e)^(step/2000)), i.e. exponential decay from 0.003->0.0001
lr = 0.0001 +  tf.train.exponential_decay(0.003, step, 2000, 1/math.e)
train_step = tf.train.AdamOptimizer(lr).minimize(cross_entropy)

# initialize
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

# make shell to plot graph for acc and loss
train_acc_list = []
test_acc_list = []
train_loss_list = []
test_loss_list = []

# iteration for counting to batchsize
iteration=0


 # make test batch
test_set=makex.make_examples_from_binvox('test')
test_input=test_set[0] 
test_label=test_set[1]
test_input_batch=[]
test_label_batch=[]
   
for temp in test_input:
    iteration +=1
    if iteration % (test_set(3)/batchsize_per_step)==0:
        test_input_batch.append(np.array(list(temp)))
iteration=0
for temp in test_label:
    iteration +=1
    if iteration%(test_set(3)/batchsize_per_step)==0:
        test_label_batch.append(np.array(list(temp)))
iteration=0

# loop for learning in num_epoch
for number_epoch in range(300) :
        
    ## generator loading ....
    train_set=makex.make_examples_from_binvox('train')
    train_input=train_set[0] 
    train_label=train_set[1]
    
    for step in range(step_per_1epoch):
        
        # make train batch and reset
        train_input_batch=[]
        train_label_batch=[]
        
    
        # train batch
        for temp in train_input:
            iteration +=1
            train_input_batch.append(np.array(list(temp)))
            if iteration == batchsize_per_step:
                iteration=0
                break
        for temp in train_label:
            iteration +=1
            train_label_batch.append(np.array(list(temp)))
            if iteration == batchsize_per_step:
                iteration=0
                break
        
        # reshape dims(x-y-z => x-y-z-channel)
        train_input_batch=np.array(train_input_batch)
        train_label_batch=np.array(train_label_batch)
        train_input_batch=train_input_batch.reshape(batchsize_per_step,DIMS[0],DIMS[1],DIMS[3],1)
        train_label_batch=train_label_batch.reshape(batchsize_per_step,number_of_class)
            
        # run    
        sess.run(train_step, {X: train_input_batch, Y_:train_label_batch, step: number_epoch, pkeep: 0.75})
            
        # put acc and loss in the shell
        a, c = sess.run([accuracy, cross_entropy],  feed_dict={X: train_input_batch, Y_: train_label_batch, pkeep: 1.0, step: number_epoch})
        print("training : ", number_epoch, ' accuracy = ', '{:7.4f}'.format(a), ' loss = ', c)        
        train_acc_list.append(a)
        train_loss_list.append(c)
            
        a, c = sess.run([accuracy, cross_entropy], feed_dict={X:  test_input_batch, Y_: test_label_batch, pkeep: 1.0})
        print("testing* : ",number_epoch, ' accuracy = ', '{:7.4f}'.format(a), ' loss = ', c)
        test_acc_list.append(a)
        test_loss_list.append(c)
   
        


# draw graph : accuracy
x = np.arange(len(train_acc_list))
plt.figure(1) 
plt.plot(x, train_acc_list,  label='train', markevery=1)
plt.plot(x, test_acc_list, label='test', markevery=1)
plt.xlabel("epochs")
plt.ylabel("accuracy")
plt.ylim(0, 1.0)
plt.legend(loc='lower right')
plt.show()

# draw graph : loss
x = np.arange(len(train_loss_list))
plt.figure(2) 
plt.plot(x, train_loss_list,  label='train', markevery=1)
plt.plot(x, test_loss_list, label='test', markevery=1)
plt.xlabel("epochs")
plt.ylabel("loss")
plt.ylim(0, 100)
plt.legend(loc='upper right')
plt.show()