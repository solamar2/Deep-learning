import numpy as np
import math as math
from section1 import *
from section2 import *
from section3 import *
def create_batches(X,Y, batch_size):
    """
    Divides the dataset into batches, including the last smaller batch if necessary.

    Input:
    X -  data, numpy array of shape (N, M), where N is the number of features and M is the number of examples
    Y -Labels, numpy array of shape (1, M)
    batch_size - Size of each batch

    Output:
    batches - List of tuples (X_batch, Y_batch)
    """
    m = X.shape[1]  # Number of examples
    batches = []

    # Shuffle the data and the labels
    rand_data_index = np.random.permutation(m)
    X_shuffled = X[:, rand_data_index]
    Y_shuffled = Y[:, rand_data_index]

    # Creating the batches
    for k in range(0, m, batch_size):
      if (k+batch_size)<m:
        X_batch = X_shuffled[:, k:k + batch_size]
        Y_batch = Y_shuffled[:, k:k + batch_size]
      else:
          X_batch = X_shuffled[:, k:m]
          Y_batch = Y_shuffled[:, k:m]
      batches.append((X_batch, Y_batch))


    return batches

def preprocess_labels(Y,num_class):
  """
  This function create a zero matrix with 1 in a specific entrance in each coloum( corresponding to the label) 
  with number of rows as the number of classes and
  the size of samples as the the number of colums.

  Input:
  Y-vector of labels with size 1xm, when m is the sample size
  num_class=how many classes i wish to classify in between

  Output:
  Y_processed- the processed matrix

  """
  # Create a zero matrix with shape (num_rows, len(vector))
  Y_processed = np.zeros((num_class, len(Y)))

  # Set the corresponding entries to 1
  for col, row in enumerate(Y):
    Y_processed[row, col] = 1
  return Y_processed

def preprocessing_data(X):
  """
  This function making the preprocessing of the dataset.
  The preprocessing including normalization and vectorization of each image

  Input:
  X-dataset
  Output:
  X_processed-the data after the preprocessing

  """
  size_image=X[0].size
  # Normalize the pixel values to be between 0 and 1
  X_Normlized= X.astype('float32') / 255.0

  # Flatten the images
  X_processed = X_Normlized.reshape(X_Normlized.shape[0], -1).T 

  return X_processed

