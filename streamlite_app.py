#!/usr/bin/env python
# coding: utf-8

# In[8]:


import torch
import torchvision
from torch import nn
import torchvision.transforms as transforms
import streamlit as st
from PIL import Image



device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



class Modified_Alexnet(nn.Module):
    """
    Implementation of a modified version of the AlexNet convolutional neural network for image classification.

    Args:
        num_classes (int): The number of output classes for the network.

    Attributes:
        conv_block (nn.Sequential): The convolutional layers of the network.
        avgpool (nn.AdaptiveAvgPool2d): The adaptive average pooling layer for the network.
        classifier (nn.Sequential): The fully connected layers of the network.

    Methods:
        forward(x): Passes input through the network and returns the output.
        predict(x): Passes input through the network, applies a softmax function, and returns the predicted class label.

    """
    def __init__(self,num_classes):
        super(Modified_Alexnet,self).__init__()
        self.num_classes=num_classes
        self.conv_block = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            
        )
        self.avgpool = nn.AdaptiveAvgPool2d((3, 3))

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(1152,4096),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(4096, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, self.num_classes),
        )

    def forward(self, x):
        x = self.conv_block(x)
        x=self.avgpool(x)
        x = self.classifier(x)
        return x
    
    def predict(self,x):
        x = self.conv_block(x)
        x=self.avgpool(x)
        x = self.classifier(x)
        out=nn.Softmax()(x)
        #print(out)
        y_pred=torch.argmax(out,axis=1).item()
        return y_pred,out


# In[ ]:


# Load pre-trained model
cnn=Modified_Alexnet(3)
cnn.load_state_dict(torch.load("model.pth", map_location=torch.device('cpu')))


# Set up device (GPU if available, otherwise CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cnn.to(device)

# Define class labels
class_labels = ['dogs', 'food','vehicle']

# Define predict function
def predict(image):
    # Transform image to tensor and normalize pixel values
    # Define the transformations to be applied to the images
    transform = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    image_tensor = transform(image).unsqueeze(0)

    # Move tensor to device
    image_tensor = image_tensor.to(device)

    # Predict class probabilities
    with torch.no_grad():
        output,prob = cnn.predict(image_tensor)
        #probabilities = torch.nn.functional.softmax(output[0], dim=0)

    # Get predicted class label
    # label_index = torch.argmax(probabilities).item()
    # predicted_label = class_labels[label_index]

    return output,prob[0,output].item()

def main():
    # Set up Streamlit app
    st.title("Image Classification App")

    # Create file uploader
    uploaded_file = st.file_uploader("Choose an image...", type="jpg")
    pred_map={0:"Dog",1:"Food",2:"Vehicle"}
    # Check if file has been uploaded
    if uploaded_file is not None:
        # Load image from uploaded file
        image = Image.open(uploaded_file)

        # Show uploaded image
        st.image(image, caption="Uploaded Image", use_column_width=True)

        # Predict image class
        predicted_label,confidence = predict(image)

        # Show predicted class label and confidence score
        st.write(f"The above image looks like {pred_map[predicted_label]} with probability of {round(confidence,2)*100} %")
        #st.write("Confidence score:", confidence)

if __name__ == "__main__":
    main()

