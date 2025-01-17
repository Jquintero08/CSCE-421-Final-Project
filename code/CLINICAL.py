#%%
import os
import torch
import torchvision
from torchvision import transforms

from torch.utils.data import WeightedRandomSampler

import matplotlib.pyplot as plt
from torchvision.utils import make_grid
import numpy as np
from sklearn import metrics

import medmnist 
from medmnist import INFO, Evaluator

from libauc.models import resnet18 as ResNet18
#from libauc.models import resnet50 as ResNet50
from libauc.losses import AUCMLoss

from torch.nn import BCELoss 
from torch.optim import SGD
from libauc.optimizers import PESG
import sys 

torch.backends.cudnn.deterministic=True
torch.backends.cudnn.benchmark=False
torch.manual_seed(2024)
if torch.cuda.is_available():
  device = torch.device("cuda")
else:
  device = torch.device("cpu")
print(f'{device=}')

np.random.seed(2024)

#---

class dataset(torch.utils.data.Dataset):
    def __init__(self, inputs, targets, trans=None):
        self.x = inputs
        self.y = targets
        self.trans=trans

    def __len__(self):
        return self.x.size()[0]

    def __getitem__(self, idx):
        if self.trans == None:
            return (self.x[idx], self.y[idx], idx)
        else:
            return (self.trans(self.x[idx]), self.y[idx])  

def train(net, train_loader, test_loader, loss_fn, optimizer, epochs):
    bestAUC = 0
    for e in range(epochs):
        net.train()
        for data, targets in train_loader:
            #print("data[0].shape: " + str(data[0].shape))
            #exit() 
            targets = targets.to(torch.float32)
            data, targets = data.to(device), targets.to(device)
            logits = net(data)
            preds = torch.flatten(torch.sigmoid(logits))
            #print("torch.sigmoid(logits):" + str(torch.sigmoid(logits)), flush=True)
            #print("preds:" + str(preds), flush=True)
            #print("targets:" + str(targets), flush=True)
            loss = loss_fn(preds, targets) 
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        #Best model saving
        currAUC = evaluate(net, test_loader, epoch=e)

        if currAUC > bestAUC:
            bestAUC = currAUC
            torch.save(net.state_dict(), f"saved_model/best_model_epoch_{e}.pth")
            print(f"Saved new best model with AUC: {bestAUC} at epoch {e}")
        
        #evaluate(net, test_loader, epoch=e)

def evaluate(net, test_loader, epoch=-1):
    # Testing AUC
    net.eval() 
    score_list = list()
    label_list = list()
    for data, targets in test_loader:
        data, targets = data.to(device), targets.to(device)
                
        score = net(data).detach().clone().cpu()
        score_list.append(score)
        label_list.append(targets.cpu()) 
    test_label = torch.cat(label_list)
    test_score = torch.cat(score_list)
                   
    testAUC = metrics.roc_auc_score(test_label, test_score)                   
    print("Epoch:" + str(epoch) + "Test AUC: " + str(testAUC), flush=True)

    #Best model saving
    return testAUC

#---
train_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Grayscale(3),
    #transforms.RandomHorizontalFlip(), #Horizontal Flip Augmentaton
    #transforms.RandomRotation(45), #45 Degree Rotation Augmentation
    transforms.Resize((32, 32)), 
    transforms.ToTensor(),
    transforms.Normalize(0.5, 0.5),
])

eval_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Grayscale(3),
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize(0.5, 0.5),
])

root = '../data'
data = sys.argv[1]
# python CLINICAL.py breastmnist
# data = 'breastmnist'
# python CLINICAL.py pneumoniamnist
# data = 'pneumoniamnist'
info = INFO[data]
DataClass = getattr(medmnist, info['python_class'])
test_dataset = DataClass(split='test', download=True, root=root)

test_data = test_dataset.imgs
test_labels = test_dataset.labels[:, 0]
    
test_labels[test_labels != 0] = 99
test_labels[test_labels == 0] = 1
test_labels[test_labels == 99] = 0

test_data = test_data/255.0
test_data = torch.tensor(test_data, dtype=torch.float32)
test_labels = torch.tensor(test_labels) 

test_dataset = dataset(test_data, test_labels, trans=eval_transform)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=128, shuffle=False, num_workers=0)

train_dataset = DataClass(split='train', download=True, root=root)

train_data = train_dataset.imgs 
train_labels = train_dataset.labels[:, 0]

train_labels[train_labels != 0] = 99
train_labels[train_labels == 0] = 1
train_labels[train_labels == 99] = 0

train_data = train_data/255.0
train_data = torch.tensor(train_data, dtype=torch.float32)
train_labels = torch.tensor(train_labels) 

train_dataset = dataset(train_data, train_labels, trans=train_transform)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=0)

net = ResNet18(pretrained=False)
net = net.to(device)
loss_fn = BCELoss()
optimizer = SGD(net.parameters(), lr=0.001,momentum=0.9,weight_decay=5e-4)

train(net,train_loader, test_loader, loss_fn, optimizer, 100)

# %%
