import streamlit as st
from PIL import Image
import torch
from torchvision import transforms
import torchvision.models as models
import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 64 * 64, 512)
        self.fc2 = nn.Linear(512, 32)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 64 * 64 * 64)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

model = SimpleCNN()
state_dict = torch.load('weights/bird_weights.pth')
model.load_state_dict(state_dict)
model.eval()

st.title('Animal Classification')
st.write('Upload an image, and the CNN will predict the species.')

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png"])
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image.', use_column_width=True)
    st.write("")
    st.write("Classifying...")

    def preprocess_image(image):
        """Resize, convert to tensor, and normalize the image."""
        transformation = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        image = transformation(image).unsqueeze(0)
        return image

    processed_image = preprocess_image(image)

    with torch.no_grad():
        prediction = model(processed_image)
        _, predicted = torch.max(prediction.data, 1)
    
    #NAIVE LABEL ASSIGNMENT, PLS FIX
    labels = {0: 'Anseriformes',
            1: 'Sphenisciformes',
            2: 'Gaviiformes',
            3: 'Passeriformes',
            4: 'Pelecaniformes',
            5: 'Cuculiformes',
            6: 'Gruiformes',
            7: 'Charadriiformes',
            8: 'Columbiformes',
            9: 'Piciformes',
            10: 'Procellariiformes',
            11: 'Caprimulgiformes',
            12: 'Accipitriformes',
            13: 'Cathartiformes',
            14: 'Opisthocomiformes',
            15: 'Suliformes',
            16: 'Trogoniformes',
            17: 'Galliformes',
            18: 'Coraciiformes',
            19: 'Psittaciformes',
            20: 'Podicipediformes',
            21: 'Ciconiiformes',
            22: 'Bucerotiformes',
            23: 'Strigiformes',
            24: 'Falconiformes',
            25: 'Struthioniformes',
            26: 'Musophagiformes',
            27: 'Phoenicopteriformes',
            28: 'Coliiformes',
            29: 'Casuariiformes',
            30: 'Otidiformes',
            31: 'Galbuliformes'}

    st.write(f'Prediction: {labels[predicted.item()]}')
