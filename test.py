# import necessary libraries
import tensorflow as tf
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F
import os
import cv2
import numpy as np
from PIL import Image

def input_prepare(img, size):
	 img = cv2.resize(img, size)
	 img = img[...,::-1]
	 img = img / 255
	 img = np.expand_dims(img, axis=0) 
	 return img 

class CNN(nn.Module):
  def __init__(self):
    super(CNN, self).__init__()
    self.conv1 = nn.Conv2d(3, 16, kernel_size=(3, 3), padding='same')
    self.pool1 = nn.MaxPool2d(kernel_size=(2, 2), stride=2)
    self.conv2 = nn.Conv2d(16, 32, kernel_size=(5, 5), padding='same')
    self.pool2 = nn.MaxPool2d(kernel_size=(2, 2), stride=2)
    self.fc1 = nn.Linear(32 * 8 * 8, 100)
    self.dropout = nn.Dropout(0.2)
    self.fc2 = nn.Linear(100, 10)

  def forward(self, x):
    x = F.relu(self.conv1(x))
    x = self.pool1(x)
    x = F.relu(self.conv2(x))
    x = self.pool2(x)
    x = x.view(-1, 32 * 8 * 8)
    x = F.relu(self.fc1(x))
    x = self.dropout(x)
    x = self.fc2(x)
    return x

# Load the trained model
model = CNN()
model.load_state_dict(torch.load('model.pt'))
model.eval()

# Define the transform for the image
transform = transforms.Compose([transforms.Resize((32, 32)),
                                transforms.ToTensor(),
                                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])

folder = 'train/yes'
for filename in os.listdir(folder):
	# Load the image
	image = Image.open(folder + '/' + filename)
	image = transform(image).unsqueeze(0)

	# Make a prediction
	output = model(image)

	# Get the class label with the highest probability
	_, prediction = torch.max(output, 1)

	## if the prediction.item()==1, it means yes (drum)
	# Print the class label
	print("Prediction: ", prediction.item())
