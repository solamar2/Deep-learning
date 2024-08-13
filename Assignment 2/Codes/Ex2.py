import numpy as np
import torch
import time
#import matplotlib.pyplot as plt
#import seaborn as sns
#from sklearn.datasets import make_regression, make_classification
import cv2 as cv
#from torch.utils.tensorboard import SummaryWriter
#writer=SummaryWriter("C:/Users/idogu/deep/work 2")
#import torchvision
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader
from torcheval.metrics import BinaryAccuracy
#import sys

"""
# defining the dataset
This class is specifc to our dataset which is constracted from files of images of different people and one text file . 
The name of the file correspond to someone's name, and inside the file each image is with the name of the specific
 name of the file with 3 zeros and  number, indicating the number of the image to this specific indevidual.
Each line in the text file has 3 or 4 elelments seperated by a tab or a space. If there are 3 elements it means we compare 
images of the same person, but if we have four elements it means that we compare images of 2 different people.
"""
class  LFWA_Dataset(Dataset):
    def __init__(self, image_dir):
       self.image_dir = image_dir
       self.samples = self.load_samples(image_dir)

    def load_samples(self, image_dir):
       with open(image_dir) as file:
           return [line.rstrip('\n').split('\t') for line in file][1:]
       
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        (img1, img2, label)=self.read_images(self.samples[idx])
        return img1, img2, label

    def path(self, name, pic_num):
        return f'C:/Users/idogu/deep/work 2/lfwa/{name}/{name}_{pic_num.zfill(4)}.jpg'
    
    def read_image(self, name, pic_num):
        #preprocessing
        image = cv.imread(self.path(name, pic_num), 0)
        image = cv.resize(image, (105, 105)) / 255
        image = np.expand_dims(image, axis=0)
        return torch.tensor(image).type(torch.float32)
    
    def read_images(self, sample):
        if len(sample) == 4:
            image1 = self.read_image(sample[0], sample[1])
            image2 = self.read_image(sample[2], sample[3])
            label = torch.tensor([0.0], dtype=torch.float32)
        else:
            image1 = self.read_image(sample[0], sample[1])
            image2 = self.read_image(sample[0], sample[2])
            label = torch.tensor([1.0], dtype=torch.float32)
        return (image1, image2, label)
    

    
    

# creating the siamse NN
class Siamse_CNN(nn.Module):

    def __init__(self,):
        super(Siamse_CNN, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=1, out_channels=64,kernel_size=(10,10), stride=1, padding='valid')
        self.pool = nn.MaxPool2d(kernel_size=(2, 2), stride=2) # Added max pooling layer
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=128,kernel_size=(7,7), stride=1, padding='valid')
        self.conv3 = nn.Conv2d(in_channels=128, out_channels=128,kernel_size=(4,4), stride=1, padding='valid')
        self.conv4 = nn.Conv2d(in_channels=128, out_channels=256,kernel_size=(4,4), stride=1, padding='valid')
        self.linear1 = nn.Linear(256 * 6 * 6, 4096)
        self.linear2 = nn.Linear(4096, 1)

    def forward(self, img1,img2):
        # NN running on the first image
        A = self.pool(F.relu(self.conv1(img1)))
        A =self.pool(F.relu(self.conv2(A)))
        A =self.pool(F.relu(self.conv3(A)))
        A = F.relu(self.conv4(A))
        A = torch.flatten(A, start_dim=1)
        A1=torch.sigmoid(self.linear1(A))
        
       
        
        # NN running on the first image
        A = self.pool(F.relu(self.conv1(img2)))
        A =self.pool(F.relu(self.conv2(A)))
        A =self.pool(F.relu(self.conv3(A)))
        A = F.relu(self.conv4(A))
        A = torch.flatten(A, start_dim=1)
        A2=torch.sigmoid(self.linear1(A))
        
        # Calculating the distance in L1
        dist=torch.abs(A1-A2)
        output=torch.sigmoid(self.linear2(dist))
        
        
        return output

### Dropout model+ more channels
class Siamse_CNN_Improved(nn.Module):

    def __init__(self,dropout_prob=0.3):
        super(Siamse_CNN_Improved, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=1, out_channels=96,kernel_size=(10,10), stride=1, padding='valid')
        self.pool = nn.MaxPool2d(kernel_size=(2, 2), stride=2) # Added max pooling layer
        self.dropout = nn.Dropout(p=dropout_prob)
        self.conv2 = nn.Conv2d(in_channels=96, out_channels=128,kernel_size=(7,7), stride=1, padding='valid')
        self.conv3 = nn.Conv2d(in_channels=128, out_channels=256,kernel_size=(4,4), stride=1, padding='valid')
        self.conv4 = nn.Conv2d(in_channels=256, out_channels=256,kernel_size=(4,4), stride=1, padding='valid')
        
        self.linear1 = nn.Linear(256 * 6 * 6, 4096)
        self.linear2 = nn.Linear(4096, 1)

    def forward(self, img1,img2):
        # NN running on the first image
        A = self.dropout(self.pool(F.relu(self.conv1(img1))))
        A =self.dropout(self.pool(F.relu(self.conv2(A))))
        A =self.dropout(self.pool(F.relu(self.conv3(A))))
        A = F.relu(self.conv4(A))
        A = torch.flatten(A, start_dim=1)
        A1=torch.sigmoid(self.linear1(A))
        
       
        
        # NN running on the first image
        A = self.dropout(self.pool(F.relu(self.conv1(img2))))
        A =self.dropout(self.pool(F.relu(self.conv2(A))))
        A =self.dropout(self.pool(F.relu(self.conv3(A))))
        A = F.relu(self.conv4(A))
        A = torch.flatten(A, start_dim=1)
        A2=torch.sigmoid(self.linear1(A))
        
        # Calculating the distance in L1
        dist=torch.pow(A1-A2,2)
        dist=(dist-torch.mean(dist))/torch.std(dist)
        output=torch.sigmoid(self.linear2(dist))
        
        
        return output



def train_model(dataloader,model,Criterion,Optimizer,num_epochs):
    """
    Input:
        dataloader: the dataset
        model: the model we use
        criterion: the loss function
        optimizer:our optimizer, usually adam.
        num_epochs: the number of epochs we use.
        
        output:
            Accuracy_vec:the accuracy along the learning of the train.
            loss_vec: the loss along the learning of the train.
            model: the model after it learned.
            
    """
    model.train()
    accuracy_metric = BinaryAccuracy()
    Current_loss=0
    accuracy=0
    Accuracy_vec=np.zeros(0)
    loss_vec=np.zeros(0)
    
    
    for num_step, (X1, X2, Y) in enumerate(dataloader):
            
        X1, X2, Y = X1.to(device), X2.to(device), Y.to(device)    
        pred=model(X1, X2)
        Loss=Criterion(pred,Y)
        Current_loss += Loss.item()
        accuracy_metric.update(pred.squeeze(), Y.squeeze()) 
            
        # Backpropagation
        Optimizer.zero_grad()
        Loss.backward()
        optimizer.step()
            

               
        if (num_step + 1) % 5 == 0:
            Average_loss = Current_loss / 5
            accuracy = accuracy_metric.compute()
            Accuracy_vec = np.append(Accuracy_vec, accuracy.cpu().numpy())
            loss_vec = np.append(loss_vec, Average_loss)
            print(f'Train: Epoch num: {num_epochs + 1}, step num: {num_step + 1}, Loss: {Average_loss}, Accuracy: {accuracy}')
            accuracy_metric.reset()
            Current_loss = 0
            
            
    # Save the model
    torch.save(model.state_dict(), 'C:/Users/idogu/deep/work 2/model.pth')
       
    return (Accuracy_vec,loss_vec,model)

def validation_run(model,dataloader,Criterion,epoch):
    """
    Input:
        dataloader: the dataset of the validation
        model: the model we use
        criterion: the loss function
        num_epochs: the number of epochs we use.
        
        output:
            Accuracy_vec:the accuracy along the learning of the validation.
            loss_vec: the loss  of the validation.
            
            
    """
    
    accuracy_metric = BinaryAccuracy()
    model.eval()
    with torch.no_grad():
        for num_step, (X1, X2, Y) in enumerate(dataloader):
            X1, X2, Y = X1.to(device), X2.to(device), Y.to(device)
            pred = model(X1,X2)
            Loss=Criterion(pred,Y)
            Loss=Loss.item()
            accuracy_metric.update(pred.squeeze(), Y.squeeze())
            validation_accuracy = accuracy_metric.compute()
            print(f'Validation: Epoch num: {epoch+1}, Loss: {Loss}, Accuracy: { validation_accuracy}')
            accuracy_metric.reset()
    return validation_accuracy.cpu().numpy(),Loss
    
def Run_experiment(model,num_epochs,validation_dataloader,train_dataloader,Criterion,Optimizer):
    
    """
    Description:
        this function running the all process of the learning on the train and on the validation set.
    Input:
        model: the model we use
        num_epochs: the number of epochs we use.
        validation_dataloader: the dataset of the validation
        train_dataloader: the dataset of the train
        criterion: the loss function
        optimizer:our optimizer, usually adam.
        
        
        output:
            Running time of the learinng process
            
    """
    Train_Accuracy=np.zeros(0)
    Train_loss=np.zeros(0)
    Valid_accuracy=np.zeros(0)
    Valid_loss=np.zeros(0)
   
    start_time = time.time()
    for epoch in range(num_epochs):
        
        # training for one epoch
        Current_train_Accuracy,Current_train_loss,model=train_model(train_dataloader,model,Criterion,Optimizer,epoch)
        
        # validation
        Current_valid_Accuracy,Current_valid_loss=validation_run(model,validation_dataloader,Criterion,epoch)
        
        # saving data for later
        Train_Accuracy = np.append(Train_Accuracy, Current_train_Accuracy)
        Train_loss = np.append(Train_loss, Current_train_loss)
        Valid_accuracy=np.append(Valid_accuracy, Current_valid_Accuracy)
        Valid_loss=np.append(Valid_loss, Current_valid_loss)
        
        if epoch > 8:
            if (Train_Accuracy[-1]>0.95  and np.mean(np.abs(np.diff(Train_loss[-18:])))<0.01 ) or ((Valid_loss[-1]-Valid_loss[-4])>0.05 and (Valid_loss[-1]-np.mean(Train_loss[-9:]))>0.3):
                end_time = time.time()
                running_time = end_time - start_time
                #save
                torch.save(model.state_dict(), 'C:/Users/idogu/deep/work 2/model.pth')
                np.save('C:/Users/idogu/deep/work 2/T_Accuracy.npy', Train_Accuracy)
                np.save('C:/Users/idogu/deep/work 2/T_Loss.npy', Train_loss)
                np.save('C:/Users/idogu/deep/work 2/V_Accuracy.npy', Valid_accuracy)
                np.save('C:/Users/idogu/deep/work 2/V_Loss.npy', Valid_loss)
                
                return print(f' Early stopping in running time of :{running_time} seconds')
    
    
    # Calculate the running time
    end_time = time.time()
    running_time = end_time - start_time
    
    #save
    np.save('C:/Users/idogu/deep/work 2/T_Accuracy.npy', Train_Accuracy)
    np.save('C:/Users/idogu/deep/work 2/T_Loss.npy', Train_loss)
    np.save('C:/Users/idogu/deep/work 2/V_Accuracy.npy', Valid_accuracy)
    np.save('C:/Users/idogu/deep/work 2/V_Loss.npy', Valid_loss)
    
    return print(running_time)


# Main
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")

 # creating the datasets using the dataLoaders 
train_dataset = LFWA_Dataset('C:/Users/idogu/deep/work 2/trainset.txt')
train_dataloader = DataLoader(train_dataset, batch_size=100, shuffle=True)
validation_dataset =  LFWA_Dataset('C:/Users/idogu/deep/work 2/validationset.txt')
validation_dataloader = DataLoader(validation_dataset, batch_size=432, shuffle=True)


#Initilaize the model
# Check if CUDA is available
if torch.cuda.is_available():
    print("CUDA is available. You have a GPU.")
    print(f"Number of available GPUs: {torch.cuda.device_count()}")
    print(f"CUDA version: {torch.version.cuda}")
    for i in range(torch.cuda.device_count()):
        print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
else:
    print("CUDA is not available. No GPU detected.")
""


"""
# Adam optimizer, without L2
model=Siamse_CNN().to(device)
# Defining hyperparameters
learning_rate=1e-4
#Defining loss function and adam optimizer with L2 regularizatrion
criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

num_epochs=30

Run_experiment(model,num_epochs,validation_dataloader,train_dataloader,criterion,optimizer)

"""
"""
# Adam optimizer, with weight decay
model=Siamse_CNN().to(device)
# Defining hyperparameters
learning_rate=1e-4
weight_decay=1e-5
#Defining loss function and adam optimizer with L2 regularizatrion
criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate,weight_decay=weight_decay)

num_epochs=30

Run_experiment(model,num_epochs,validation_dataloader,train_dataloader,criterion,optimizer)
"""


#model with dropout
model = Siamse_CNN_Improved(dropout_prob=0.3)
model=model.to(device)
learning_rate=1e-4
weight_decay=1e-5
#Defining loss function and adam optimizer with L2 regularizatrion
criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate,weight_decay=weight_decay)

num_epochs=30

Run_experiment(model,num_epochs,validation_dataloader,train_dataloader,criterion,optimizer)
