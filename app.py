import torch
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader, ConcatDataset
import random
import sys
import torch.nn as nn
import torch.nn.functional as f
from sklearn.metrics import confusion_matrix, accuracy_score
import numpy as np
import glob
import matplotlib.pyplot as plt
import cv2
from PIL import Image, ImageOps
from torchvision.transforms import ToTensor


from flask import Flask, render_template, request

app = Flask(__name__)

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.cnn_model = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=6, kernel_size=5),
            nn.Tanh(),
            nn.AvgPool2d(kernel_size=2, stride=5),
            nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5),
            nn.Tanh(),
            nn.AvgPool2d(kernel_size=2, stride=5)
        )
        self.fc_model = nn.Sequential(
            nn.Linear(in_features=256, out_features=120),
            nn.Tanh(),
            nn.Linear(in_features=120, out_features=84),
            nn.Tanh(),
            nn.Linear(in_features=84, out_features=1)
        )

    def forward(self, x):
        x = self.cnn_model(x)
        x = x.reshape(x.size(0), -1)
        x = self.fc_model(x)
        x = torch.sigmoid(x)
        return x

# Create an instance of your custom CNN

device = torch.device('cpu')
model = CNN().to(device)
state_dict = torch.load("./neuro_nova/last_lo.pth")
model.load_state_dict(state_dict)
model.eval()
transform = ToTensor()

 # Define the image transformation for PyTorch

@app.route('/', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        imagefile = request.files['imagefile']
        image_path = "./images/" + imagefile.filename
        imagefile.save(image_path)

        # Open and preprocess the image using OpenCV
        img = cv2.imread(image_path)
        img = cv2.resize(img, (128, 128))
        b, g, r = cv2.split(img)
        img = cv2.merge([r, g, b])
        img = img / 255.0  # Normalize the image

        # Convert the image to a PyTorch tensor
        image_tensor = transform(img).unsqueeze(0)
        # Convert the input tensor to float
        image_tensor = image_tensor.float()

        # Perform inference on the image using the model
        with torch.no_grad():
            output = model(image_tensor)
        print(output)
        # Process the output (e.g., convert logits to class probabilities)

        value = output[0][0].item()
        result=""
        if value >=0.5:
            result+="this is tumor "
        else:
            result+="not a tumor"
        # predicted_class = output.argmax(dim=1).item()
        #
        # print(predicted_class)
        return render_template('index.html',prediction=result)
    else:
        return render_template('index.html')
if __name__ == '__main__':
    app.run(debug=True)