import os
import torch
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm


class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 64 * 64, 512)
        self.fc2 = nn.Linear(512, len(set(dataset.labels)))

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 64 * 64 * 64)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    

class CustomDataset(Dataset):
    def __init__(self, root_dir, transform=None, allowed_extensions=('jpg', 'jpeg', 'png', 'bmp', 'gif')):
        self.root_dir = root_dir
        self.transform = transform
        self.images = []
        self.labels = []
        self.class_to_idx = {}

        for dir_name, _, file_names in os.walk(root_dir):
            if file_names:
                parts = dir_name.split(os.sep)[-1].split('_')
                if len(parts) >= 4 and parts[3] == 'Aves':
                    label = parts[4]
                    if label not in self.class_to_idx:
                        self.class_to_idx[label] = len(self.class_to_idx)
                    label_index = self.class_to_idx[label]
                    for file_name in file_names:
                        if file_name.split('.')[-1].lower() in allowed_extensions:
                            self.images.append(os.path.join(dir_name, file_name))
                            self.labels.append(label_index)
                            print(f"Added {file_name} with label {label}")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = self.images[idx]
        image = Image.open(img_name).convert('RGB')
        label = self.labels[idx]
        
        if self.transform:
            image = self.transform(image)

        return image, label


transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

dataset_path = 'train_mini'
dataset = CustomDataset(root_dir=dataset_path, transform=transform)
train_loader = DataLoader(dataset, batch_size=32, shuffle=True)

model = SimpleCNN()

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

for epoch in range(2):  
    running_loss = 0.0
    for i, (inputs, labels) in enumerate(tqdm(train_loader, desc=f'Epoch {epoch+1}', leave=False)):
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        if (i + 1) % 100 == 0:
            print(f'[{epoch + 1}, {i + 1}] loss: {running_loss / 100:.3f}')
            running_loss = 0.0

    tqdm.write(f'Completed Epoch {epoch+1}')

model_path = 'weights/bird_weights.pth'
torch.save(model.state_dict(), model_path)