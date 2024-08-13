"""
This code implement the models on the testset
"""

# importing libraries
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
    

    
    

# CNN artitacture
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
    
    
# Main
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")

test_dataset = LFWA_Dataset('C:/Users/idogu/deep/work 2/pairsDevTest.txt')
test_dataloader = DataLoader(test_dataset, batch_size=1000, shuffle=True)
criterion = nn.BCELoss()
accuracy_metric = BinaryAccuracy()
correct_predictions = []
incorrect_predictions = []

# Load the model using the same class definition
model = Siamse_CNN()
model.load_state_dict(torch.load('C:/Users/idogu/deep/work 2/dropout_weight/model.pth'))
model.to(device)
model.eval()  # Set the model to evaluation mode

for num_step, (X1, X2, Y) in enumerate(test_dataloader):
    X1, X2, Y = X1.to(device), X2.to(device), Y.to(device)
    pred = model(X1, X2)
    
  
accuracy_metric.update(pred.squeeze(), Y.squeeze()) 
accuracy=accuracy_metric.compute()
accuracy=accuracy.item()
a = 1
    
    