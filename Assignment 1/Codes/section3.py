import numpy as np
import math as math
from section1 import *





def Predict(X, Y, parameters):
  """
  Description:
The function receives an input data and the true labels and calculates the accuracy of the trained neural network on the data.

Input:
X – the input data, a numpy array of shape (height*width, number_of_examples)
Y – the “real” labels of the data, a vector of shape (num_of_classes, number of examples)
Parameters – a python dictionary containing the DNN architecture’s parameters

Output:
accuracy – the accuracy measure of the neural net on the provided data (i.e. the percentage of the samples for which the correct label receives the hughest confidence score). Use the softmax function to normalize the output values.
  """
  from sklearn.metrics import accuracy_score
  use_batchnorm='False'
  AL,_=L_model_forward(X, parameters, use_batchnorm)
  Y_hat = np.argmax(AL, axis=0)
  Y_true = np.argmax(Y, axis=0)
  """
  Accuracy= np.sum(Y_hat==Y_true)/len(Y_hat)
  """
  Accuracy=accuracy_score(Y_true,Y_hat)

  return Accuracy
