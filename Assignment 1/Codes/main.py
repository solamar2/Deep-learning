
import numpy as np
import math as math
import keras
from keras.datasets import mnist
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from section1 import *
from section2 import *
from section3 import *
from section4 import *


def L_layer_model(X, Y, layers_dims, learning_rate, num_iterations, batch_size):
  """
Description:
Implements a L-layer neural network. All layers but the last should have the ReLU activation function, and the final layer will apply the softmax activation function. The size of the output layer should be equal to the number of labels in the data. Please select a batch size that enables your code to run well (i.e. no memory overflows while still running relatively fast).

Input:
X – the input data, a numpy array of shape (height*width , number_of_examples)
Comment: since the input is in grayscale we only have height and width, otherwise it would have been height*width*3
Y – the “real” labels of the data, a vector of shape (num_of_classes, number of examples)
Layer_dims – a list containing the dimensions of each layer, including the input
batch_size – the number of examples in a single training batch.

Output:
parameters – the parameters learnt by the system during the training (the same parameters that were updated in the update_parameters function).
costs – the values of the cost function (calculated by the compute_cost function). One value is to be saved after each 100 training iterations (e.g. 3000 iterations -> 30 values).


  """
  use_batchnorm='False'# should be change
  costs=[]
  iter=0
  num_batch=math.floor(X.shape[1]/batch_size)
  num_epoch=math.floor(num_iterations/num_batch)
  


  # initialize parameters
  parameters=initialize_parameters(layers_dims)

  for epoch in range(1, num_epoch+1):
    #Seperating into batches, randomly for each epoch
    batches=create_batches(X,Y, batch_size)
    for i in range(0,len(batches)):
      iter=iter+1
      X_batch, Y_batch = batches[i]
      #running the forward propagation
      AL,caches=L_model_forward(X_batch, parameters, use_batchnorm)
      #computing the cost
      cost=compute_cost(AL, Y_batch)
      if iter % 100 ==0:
        costs.append(cost)
      #running the backpropagation
      grads=L_model_backward(AL, Y_batch, caches)
      # updating the parameter
      parameters=Update_parameters(parameters, grads, learning_rate)


  return parameters,costs,iter

if __name__ == '__main__':

    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

    #spliting the train to validation and train
    x_train, x_validation, y_train, y_validation = train_test_split(x_train, y_train, test_size=0.2, random_state=42)
    #preprocessing
    x_train=preprocessing_data(x_train)
    y_train=preprocess_labels(y_train,10)
    x_validation=preprocessing_data(x_validation)
    y_validation=preprocess_labels(y_validation,10)
    x_test=preprocessing_data(x_test)
    y_test=preprocess_labels(y_test,10)
    
    
    # difining parameters
    layers_dims=[784,20,7,5,10]
    learning_rate=0.009
    batch_size=256# power of 2
    num_iterations=100
    #Training the model
    parameters,costs,iter=L_layer_model(x_train, y_train, layers_dims, learning_rate, 10000, batch_size)
    print(3)
    print(4)