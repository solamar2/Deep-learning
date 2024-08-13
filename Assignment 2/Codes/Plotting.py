import numpy as np
import matplotlib.pyplot as plt
import os
# Load the NumPy file
data_train = np.load('C:/Users/idogu/deep/work 2/normal/T_Accuracy.npy')
data_valid = np.load('C:/Users/idogu/deep/work 2/normal/V_Accuracy.npy')

Loss_train = np.load('C:/Users/idogu/deep/work 2/normal/T_Loss.npy')
Loss_valid = np.load('C:/Users/idogu/deep/work 2/normal/V_Loss.npy')
# Calculate the number of epochs (make sure it's an integer)
num_epoch = int(len(data_train) / 3)

# Initialize the epoch array
epoch_train = np.zeros(0)
epoch_valid=np.arange(1, num_epoch+1)
# Populate the epoch array
for i in range(num_epoch):
    epoch_train = np.append(epoch_train, [i + 0.3, i + 0.7, i + 1])


# Create the subplot structure (1 row, 2 columns)
fig, axes = plt.subplots(1, 2, figsize=(12, 6))

# Plotting on the first subplot
axes[0].plot(epoch_train, data_train)
axes[0].plot(epoch_valid, data_valid)

axes[0].set_xlabel('Epoch')  
axes[0].set_ylabel('Accuracy')
axes[0].legend(['Train','Validation'])
axes[0].grid()

axes[1].plot(epoch_train, Loss_train)
axes[1].plot(epoch_valid, Loss_valid)

axes[1].set_xlabel('Epoch')  
axes[1].set_ylabel('Loss') 
axes[1].grid()

fig.suptitle('Siamse CNN', fontsize=20)


file_path = os.path.join('C:/Users/idogu/deep/work 2/normal/normal_model.png')
plt.savefig(file_path)
plt.show()