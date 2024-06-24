import torch
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms
import torch.nn as nn
import torch.optim as optim
import os
from glob import glob
from tqdm import tqdm
from PIL import Image, ImageChops, ImageEnhance
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
import torch.nn.functional as F
import matplotlib.pyplot as plt 
import random

# Dosya yolları
image_dir = r"C:\Users\ceren\OneDrive\Desktop\bitirme\flask3\static\images\CASIA2 Dataset"
ela_dir = r"C:\Users\ceren\OneDrive\Desktop\bitirme\flask3\static\images\CASIA2_ELA"

# Tüm dosyaları bulma
total_files = glob(os.path.join(image_dir, "**", "*"), recursive=True)
print('total files', len(total_files))

# Dosya türlerini belirleme
types = set(file.split(".")[-1] for file in total_files)
print('types of files in the folder', types)

# jpg dosyalarını bulma
jpg_files = glob(os.path.join(image_dir, "**", "*.jpg"), recursive=True)
print('jpg files: ', len(jpg_files))

# tif dosyalarını bulma
tif_files = glob(os.path.join(image_dir, "**", "*.tif"), recursive=True)
print('tif files: ', len(tif_files))

# Tp_jpg_files
tp_jpg_files = glob(os.path.join(image_dir, "**", "Tp*"), recursive=True)
print('tp_jpg_files: ', len(tp_jpg_files))
print('first 5 tp_jpg_files: ', tp_jpg_files[:5])

def convert_to_ela_image(image_path, quality=90):
    temp_file = 'temp.jpg'
    im = Image.open(image_path).convert('RGB')
    im.save(temp_file, 'JPEG', quality=quality)
    saved = Image.open(temp_file).convert('RGB')
    original = Image.open(image_path).convert('RGB')

    if original.size != saved.size:
        saved = saved.resize(original.size)

    diff = ImageChops.difference(original, saved)
    extrema = diff.getextrema()
    max_diff = max([ex[1] for ex in extrema])
    scale = 255.0 / max_diff
    diff = ImageEnhance.Brightness(diff).enhance(scale)
    os.remove(temp_file)
    return diff

images = jpg_files + tif_files
print('total images', len(images))

# Dizin oluşturma
os.makedirs(ela_dir, exist_ok=True)

# ELA görüntülerine dönüştürme ve kaydetme
for image in tqdm(images):
    try:
        ela_image = convert_to_ela_image(image)
        ela_image.save(os.path.join(ela_dir, os.path.basename(image)))
    except Exception as e:
        print(f"Error processing {image}: {e}")

# ELA dosyalarının sayısını kontrol etme
total_ela_files = glob(os.path.join(ela_dir, "*"))
print('total ela files', len(total_ela_files))
types = set(file.split(".")[-1] for file in total_ela_files)
print('types of files in the folder', types)

# TP ve AU dosyalarını sayma
def find_tp_files(files):
    tp_files, au_files = 0, 0
    for file in files:
        filename = os.path.basename(file)
        if filename.startswith('Tp'):
            tp_files += 1
        elif filename.startswith('Au'):
            au_files += 1
    return tp_files, au_files

tp_files, au_files = find_tp_files(total_ela_files)
print(f'TP files: {tp_files}, AU files: {au_files}')

# Sınıf sayısını sayma
def count_classes(dataloader):
    tempered, original = 0, 0
    for images, labels in tqdm(dataloader):
        for label in labels:
            if label == 1:
                tempered += 1
            else:
                original += 1
    print(f"Tempered: {tempered}, Original: {original}")

# Veri kümesi sınıfı
class CASIA2_ELA(Dataset):
    def __init__(self, images, transform=None):
        self.images = images
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = os.path.basename(image).split("_")[0]
        label = 1 if label == 'Tp' else 0
        image = Image.open(image)
        if self.transform:
            image = self.transform(image)
        return image, torch.tensor(label, dtype=torch.float32)

# Veri kümesini test etme
def test():
    all_images = glob(os.path.join(ela_dir, "*"))
    random.shuffle(all_images)
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
    dataset = CASIA2_ELA(all_images, transform=transform)
    print(dataset[1][0].shape)
    print(dataset[1][1])

# if __name__ == "__main__":
#     test()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

all_images = glob(os.path.join(ela_dir, "*"))
random.shuffle(all_images)

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])
dataset = CASIA2_ELA(all_images, transform=transform)

def split_dataset(dataset, train_size=0.8):
    # get the indices of the images
    indices = list(range(len(dataset)))
    # get the labels of the images
    labels = [dataset[i][1].item() for i in indices]
    # split the indices into train and validation indices
    train_indices, val_indices = train_test_split(indices, train_size=train_size, stratify=labels)
    # create the train and validation subsets
    train_dataset = Subset(dataset, train_indices)
    val_dataset = Subset(dataset, val_indices)
    return train_dataset, val_dataset

train_ds, valid_ds = split_dataset(dataset)
train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
val_loader = DataLoader(valid_ds, batch_size=32, shuffle=False)

count_classes(train_loader)
count_classes(val_loader)

def visualize(dataloader):
    for images, labels in dataloader:
        print(images.shape)
        print(labels.shape)
        plt.figure(figsize=(16, 8))
        for i in range(32):
            plt.subplot(4, 8, i+1)
            plt.imshow(images[i].permute(1, 2, 0))
            plt.title(f"Label: {labels[i]}")
            plt.axis("off")
        plt.show()
        break
visualize(train_loader)

class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(64 * 54 * 54, 128)
        self.fc2 = nn.Linear(128, 1)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        return torch.sigmoid(x)

model = CNN()
model.to(device)

criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

train_sub = Subset(train_ds, range(2000))
train_sub_dl = DataLoader(train_sub, batch_size=32, shuffle=True)

losses = []
accuracy = []

for epch in range(5):
    for xb, yb in tqdm(train_sub_dl):
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()
        yb_ = model(xb)
        loss = criterion(yb_.squeeze(1), yb)
        losses.append(loss.item())
        batch_acc = f1_score(yb.cpu().detach().numpy(), yb_.cpu().detach().numpy().round())
        accuracy.append(batch_acc)
        loss.backward()
        optimizer.step()

plt.figure(figsize=(20, 10))
plt.plot(losses)

plt.figure(figsize=(20, 10))
plt.plot(accuracy)

def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs, device):
    train_loss, val_loss, train_acc, val_acc = [], [], [], []
    for epoch in range(num_epochs):
        model.train()
        for xb, yb in tqdm(train_loader):
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            yb_ = model(xb)
            loss = criterion(yb_.squeeze(1), yb)
            loss.backward()
            optimizer.step()
            train_loss.append(loss.item())
            batch_acc = accuracy_score(yb.cpu().detach().numpy(), yb_.cpu().detach().numpy().round())
            train_acc.append(batch_acc)
        model.eval()
        with torch.no_grad():
            for xb, yb in tqdm(val_loader):
                xb, yb = xb.to(device), yb.to(device)
                yb_ = model(xb)
                loss = criterion(yb_.squeeze(1), yb)
                val_loss.append(loss.item())
                batch_acc = accuracy_score(yb.cpu().detach().numpy(), yb_.cpu().detach().numpy().round())
                val_acc.append(batch_acc)
        print(f'Epoch: {epoch+1}, Train Loss: {torch.tensor(train_loss).mean():.4f}, Train Accuracy: {torch.tensor(train_acc).mean():.4f}, Val Loss: {torch.tensor(val_loss).mean():.4f}, Val Accuracy: {torch.tensor(val_acc).mean():.4f}')

# Model eğitimi
train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=5, device=device)

# Eğitilmiş modeli kaydetme
PATH = "cnn.pth"
torch.save(model.state_dict(), PATH)

device = torch.device('cpu')
model = CNN()
model.to(device)
model.load_state_dict(torch.load(PATH, map_location=device))

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

def predict_image(image_path, model):
    img = convert_to_ela_image(image_path)
    img = transform(img)
    img = img.unsqueeze(0)
    model.eval()
    with torch.no_grad():
        output = model(img)
        _, pred = torch.max(output, 1)
    return 'fake' if pred.item() == 1 else 'real'

# Örnek tahmin
img1 = total_ela_files[0]
result = predict_image(img1, model)
print(result)

# ELA dönüşümü fonksiyonu
def convert_to_ela_image(image_path, quality=90):
    temp_file = 'temp.jpg'
    im = Image.open(image_path).convert('RGB')
    im.save(temp_file, 'JPEG', quality=quality)
    saved = Image.open(temp_file).convert('RGB')
    original = Image.open(image_path).convert('RGB')

    if original.size != saved.size:
        saved = saved.resize(original.size)

    diff = ImageChops.difference(original, saved)
    extrema = diff.getextrema()
    max_diff = max([ex[1] for ex in extrema])
    scale = 255.0 / max_diff
    diff = ImageEnhance.Brightness(diff).enhance(scale)
    os.remove(temp_file)
    return diff

# Model mimarisi
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(64 * 54 * 54, 128)
        self.fc2 = nn.Linear(128, 1)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        return torch.sigmoid(x)

# Veri kümesi sınıfı
class CASIA2_ELA(Dataset):
    def __init__(self, images, transform=None):
        self.images = images
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = os.path.basename(image).split("_")[0]
        label = 1 if label == 'Tp' else 0
        image = Image.open(image)
        if self.transform:
            image = self.transform(image)
        return image, torch.tensor(label, dtype=torch.float32)

# Veri kümesini bölme
def split_dataset(dataset, train_size=0.8):
    indices = list(range(len(dataset)))
    labels = [dataset[i][1].item() for i in indices]
    train_indices, val_indices = train_test_split(indices, train_size=train_size, stratify=labels)
    train_dataset = Subset(dataset, train_indices)
    val_dataset = Subset(dataset, val_indices)
    return train_dataset, val_dataset

# Eğitim fonksiyonu
def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs, device):
    train_loss, val_loss, train_acc, val_acc = [], [], [], []
    for epoch in range(num_epochs):
        model.train()
        for xb, yb in tqdm(train_loader):
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            yb_ = model(xb)
            loss = criterion(yb_.squeeze(1), yb)
            loss.backward()
            optimizer.step()
            train_loss.append(loss.item())
            batch_acc = accuracy_score(yb.cpu().detach().numpy(), yb_.cpu().detach().numpy().round())
            train_acc.append(batch_acc)
        
        model.eval()
        with torch.no_grad():
            for xb, yb in tqdm(val_loader):
                xb, yb = xb.to(device), yb.to(device)
                yb_ = model(xb)
                loss = criterion(yb_.squeeze(1), yb)
                val_loss.append(loss.item())
                batch_acc = accuracy_score(yb.cpu().detach().numpy(), yb_.cpu().detach().numpy().round())
                val_acc.append(batch_acc)
        
        print(f'Epoch: {epoch+1}, Train Loss: {torch.tensor(train_loss).mean():.4f}, Train Accuracy: {torch.tensor(train_acc).mean():.4f}, Val Loss: {torch.tensor(val_loss).mean():.4f}, Val Accuracy: {torch.tensor(val_acc).mean():.4f}')

# Ana kod kısmı
if __name__ == "__main__":
    image_dir = r"C:\Users\ceren\OneDrive\Desktop\bitirme\flask3\static\images\CASIA2 Dataset"
    ela_dir = r"C:\Users\ceren\OneDrive\Desktop\bitirme\flask3\static\images\CASIA2_ELA"
    
    jpg_files = glob(os.path.join(image_dir, "**", "*.jpg"), recursive=True)
    tif_files = glob(os.path.join(image_dir, "**", "*.tif"), recursive=True)
    images = jpg_files + tif_files

    os.makedirs(ela_dir, exist_ok=True)
    
    for image in tqdm(images):
        try:
            ela_image = convert_to_ela_image(image)
            ela_image.save(os.path.join(ela_dir, os.path.basename(image)))
        except Exception as e:
            print(f"Error processing {image}: {e}")
    
    all_images = glob(os.path.join(ela_dir, "*"))
    random.shuffle(all_images)
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
    
    dataset = CASIA2_ELA(all_images, transform=transform)
    train_ds, val_ds = split_dataset(dataset)
    
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=32, shuffle=False)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = CNN().to(device)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=5, device=device)
    
    PATH = "cnn.pth"
    torch.save(model.state_dict(), PATH)
    # Eğitilmiş modeli kaydetme


