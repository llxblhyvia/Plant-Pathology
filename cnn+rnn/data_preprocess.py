# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import os
import cv2
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from PIL import Image
from torchvision import transforms

csv_path = 'train.csv'
data_dir = '/nfsshare/home/dl03/cnn+rnn/train_images'
height, width = 300, 400

data = pd.read_csv(csv_path)
data['labels'] = data['labels'].apply(lambda x: x.split(' '))


lab2idx = {}
idx2lab = {}
n_label = 0
for labs in data['labels']:
    for lab in labs:
        if lab in lab2idx.keys():
            continue
        else:
            lab2idx[lab] = n_label
            idx2lab[n_label] = lab
            n_label += 1

def one_hots(labs, dict= lab2idx, n_class= n_label):
    label = np.zeros(n_class)
    for lab in labs:
        idx = dict[lab]
        label[idx] = 1
    return label

data['labels'] = data['labels'].apply(lambda x: one_hots(x))
img_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()])

def loader(path):
    try:
        img = Image.open(path).convert('RGB')
        return img_transforms(img)
    except:
        print("cannot read image: {}".format(path))

class myDataset(Dataset):
    def __init__(self, image, label, loader= loader):
        self.images = image
        self.targets = label
        self.loader = loader
    
    def __getitem__(self, index):
        path = self.images[index]
        img = self.loader(os.path.join(data_dir, path))
        target = self.targets[index]
        return img, target
    
    def __len__(self):
        return len(self.images)

batch_size = 32

def dataloader(batch_size= batch_size):
    train_imgs, valid_imgs, train_tags, valid_tags = train_test_split(data['image'].values, data['labels'].values, test_size= 0.3 )
    train_dataset, valid_dataset = myDataset(train_imgs.tolist(), train_tags.tolist()), myDataset(valid_imgs.tolist(), valid_tags.tolist())
    train_dataloader, valid_dataloader = DataLoader(train_dataset, batch_size= batch_size), DataLoader(valid_dataset, batch_size= batch_size)
    return train_dataloader, valid_dataloader
