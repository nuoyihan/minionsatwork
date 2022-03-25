import numpy as np
import matplotlib.pyplot as plt
import random
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from tqdm.auto import tqdm

def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
seed_everything(42)

def load_raw_bitmap(f):
    return torch.tensor(np.load(f).reshape(-1, 1, 28, 28), dtype=torch.float)


basketball = load_raw_bitmap("sketch-dataset/basketball.npy")
bear = load_raw_bitmap("sketch-dataset/bear.npy")
skateboard = load_raw_bitmap("sketch-dataset/skateboard.npy")


class ImageDataset(Dataset):
    def __init__(self, images, labels):
        self.images = images
        self.labels = labels

    def __len__(self):
        return len(self.images)

    def __getitem__(self, i):
        return self.images[i], self.labels[i]


def construct_datasets(image_classes):
    images = []
    labels = []
    for i, image_class in enumerate(image_classes):
        for image in image_class:
            images.append(image.clone())
            labels.append(i)

    train_indices = set(random.sample(
        range(len(images)), k=round(0.9 * len(images))))

    train_images, train_labels = [], []
    test_images, test_labels = [], []
    for i, (image, label) in enumerate(zip(images, labels)):
        if i in train_indices:
            train_images.append(image)
            train_labels.append(label)
        else:
            test_images.append(image)
            test_labels.append(label)

    return ImageDataset(train_images, train_labels), ImageDataset(test_images, test_labels)


train_dataset, test_dataset = construct_datasets(
    (basketball, bear, skateboard))

train_dataloader = DataLoader(train_dataset,
                              batch_size=128,
                              shuffle=True,
                              num_workers=16)

test_dataloader = DataLoader(test_dataset,
                             batch_size=128,
                             shuffle=False,
                             num_workers=16)


class SketchNet(nn.Module):
    def __init__(self):
        super(SketchNet, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=16,
                kernel_size=5,
                stride=1,
                padding=2,
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 32, 5, 1, 2),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.out = nn.Linear(32 * 7 * 7, 3)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0), -1)
        output = self.out(x)
        return output


sn = SketchNet()
optimizer = optim.Adam(sn.parameters(), lr=3e-4)


def train(model, dataloader, optimizer):
    model.train()
    for images, labels in tqdm(dataloader):
        logits = sn(images)
        loss = F.cross_entropy(logits, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


def acc(preds, labels):
    return sum(1 if p == l else 0 for p, l in zip(preds, labels)) / len(preds)


def test(model, dataloader):
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for images, labels in tqdm(dataloader):
            logits = sn(images)
            preds = logits.argmax(dim=1)
            all_preds.extend(preds.tolist())
            all_labels.extend(labels.tolist())

    print("acc:", acc(all_preds, all_labels))


train(sn, train_dataloader, optimizer)
test(sn, test_dataloader)

torch.onnx.export(sn,                        # model being run
                  torch.rand(1, 1, 28, 28),  # model input (or a tuple for multiple inputs)
                  "models/sketch-net.onnx",         # where to save the model (can be a file or file-like object)
                  export_params=True,        # store the trained parameter weights inside the model file
                  opset_version=10,          # the ONNX version to export the model to
                  do_constant_folding=True,  # whether to execute constant folding for optimization
                  input_names=['images'],    # the model's input names
                  output_names=['logits'],   # the model's output names
                  dynamic_axes={'images': {0: 'batch_size'},    # variable length axes
                                'logits': {0: 'batch_size'}})
