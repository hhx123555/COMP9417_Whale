import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder

import torch
from torch.utils.data import TensorDataset, DataLoader, Dataset
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data.sampler import SubsetRandomSampler
import cv2
from pytorchtools import EarlyStopping
import torch.optim.lr_scheduler as lr_scheduler

from torch.utils.data import DataLoader, random_split
import torch.nn as nn
import torch.optim as optim
from torchvision.models import resnet50

train_df = pd.read_csv("./train.csv")

train_df.head()

# Distribution of images per whale
train_df["Id"].value_counts()

count_df = train_df.groupby("Id").count().rename(columns={"Image": "Number_of_images_id"})
count_df.loc[count_df["Number_of_images_id"] > 73, "Number_of_images_id"] = 74
plt.figure(figsize=(20, 10))
sns.countplot(data=count_df, x="Number_of_images_id")
plt.title("Number of images id distribution")

fig = plt.figure(figsize=(12, 6))
for idx, name in enumerate(train_df[train_df['Id'] == 'new_whale']['Image'][:8]):
    ax = fig.add_subplot(2, 4, idx + 1)
    img = cv2.imread(os.path.join("./train", name))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    ax.imshow(img)
plt.show()

fig = plt.figure(figsize=(12, 6))
for idx, name in enumerate(train_df['Id'].value_counts().index[-8:]):
    for _, path in enumerate(train_df[train_df['Id'] == name]['Image'][:1]):
        ax = fig.add_subplot(2, 4, idx + 1)
        img = cv2.imread(os.path.join("./train", path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        ax.imshow(img)
plt.show()

len(train_df['Id'].unique())

label_encoder = LabelEncoder()
integer_encoded = label_encoder.fit_transform(train_df['Id'])

onehot_encoder = OneHotEncoder(sparse=False)
integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
y = onehot_encoder.fit_transform(integer_encoded)
y[:2]

# Data preprocessing
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomAffine(degrees=30),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


class WhaleDataLoader(Dataset):
    def __init__(self, image_path, process='train', df=None, transform=None, y=None):
        self.image_path = image_path
        self.imgs = [img for img in os.listdir(image_path)]
        self.process = process
        self.transform = transform
        self.y = y
        self.df = df

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        if self.process == 'train':
            img_name = os.path.join(self.image_path, self.df.values[idx][0])
            label = self.y[idx]

        elif self.process == 'test':
            img_name = os.path.join(self.image_path, self.imgs[idx])

        img = Image.open(img_name).convert('RGB')
        img = self.transform(img)
        if self.process == 'train':
            return img, label
        elif self.process == 'test':
            return img


train_dataloader = WhaleDataLoader(image_path="./train",
                                   process='train', df=train_df, transform=train_transform, y=y)
test_dataloader = WhaleDataLoader(image_path="./test",
                                  process='test', transform=test_transform)

train_sampler = SubsetRandomSampler(list(range(len(os.listdir("./train")))))

train_loader = torch.utils.data.DataLoader(train_dataloader, batch_size=32, sampler=train_sampler, num_workers=0)

test_loader = torch.utils.data.DataLoader(test_dataloader, batch_size=32, num_workers=0)

# ResNet50-based model
train_dataset_size = int(len(train_dataloader) - 1000)
val_dataset_size = len(train_dataloader) - train_dataset_size
train_dataset, val_dataset = random_split(train_dataloader, [train_dataset_size, val_dataset_size])

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=0)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=0)

# MODEL    resnet50
learning_rate = 0.01
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = resnet50(pretrained=True)
num_classes = len(train_df['Id'].unique())
model.fc = nn.Linear(model.fc.in_features, num_classes)
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)

# learning rate decay
scheduler = lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

early_stopping = EarlyStopping(patience=10, verbose=True, delta=0.001)


def train(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0

    for i, (inputs, labels) in enumerate(dataloader):
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, torch.argmax(labels, dim=1))

        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if i % 100 == 99:  # output loss every 100 mini-batches
            print('[Epoch %d, Batch %d] loss: %.3f' % (epoch + 1, i + 1, running_loss / 100))
            running_loss = 0.0

    return running_loss / len(dataloader)


def validate(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = len(dataloader.dataset)

    for inputs, labels in dataloader:
        inputs, labels = inputs.to(device), labels.to(device)
        with torch.no_grad():
            outputs = model(inputs)

            loss = criterion(outputs, torch.argmax(labels, dim=1))
            running_loss += loss.item()

            # Calculate top-5 accuracy
            maxk = max((1, 5))
            labels = torch.argmax(labels, dim=1)
            _, pred = outputs.topk(maxk, 1, True, True)
            correct += torch.eq(pred, labels.view(-1, 1)).sum().float().item()

            # total += labels.size(0)
            accuracy = correct / total

    return running_loss / len(dataloader), accuracy


num_epochs = 10

for epoch in range(num_epochs):

    train_loss = train(model, train_loader, criterion, optimizer, device)
    scheduler.step()
    val_loss, val_accuracy = validate(model, val_loader, criterion, device)

    # Early-Stopping
    early_stopping(val_loss, model)

    if early_stopping.early_stop:
        print("Early stopping")
        break

    print(
        f"Epoch: {epoch + 1}, Train Loss: {train_loss}, Validation Loss: {val_loss},Validation Accuracy: {val_accuracy}")


# predict test dataset
def predict(model, dataloader, device):
    model.eval()
    predictions = []

    with torch.no_grad():
        for inputs in dataloader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            predictions.extend(preds.tolist())

    return predictions


test_preds = predict(model, test_loader, device)
preds_whale_ids = label_encoder.inverse_transform(test_preds)
submission_df = pd.DataFrame({'Image': test_dataloader.imgs, 'Id': preds_whale_ids})
submission_df.to_csv('submission.csv', index=False)
