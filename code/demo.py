import os
import torch
import torchvision
from torchvision import transforms

from torch.utils.data import WeightedRandomSampler

import matplotlib.pyplot as plt
from torchvision.utils import make_grid




import numpy as np
from arguments import args
from sklearn import metrics
torch.backends.cudnn.deterministic=True
torch.backends.cudnn.benchmark=False
torch.manual_seed(2024)
if torch.cuda.is_available():
  device = torch.device("cuda")
else:
  device = torch.device("cpu")

np.random.seed(2024)

if args.server == "grace":
    os.environ['http_proxy'] = '10.73.132.63:8080'
    os.environ['https_proxy'] = '10.73.132.63:8080'
elif args.server == "faster":
    os.environ['http_proxy'] = '10.72.8.25:8080'
    os.environ['https_proxy'] = '10.72.8.25:8080'

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

def imshow(inp, title=None):
    """Imshow for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0))
    mean = 0.5
    std = 0.5
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp, cmap='gray')
    if title is not None:
        plt.title(title)
    plt.pause(0.001)


def main_worker():

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

    import medmnist 
    from medmnist import INFO, Evaluator
    #root = '/scratch/group/optmai/zhishguo/med/'
    # root = 'C:/Users/Jakey/Desktop/Spring2024/CSCE421/Final-Project/data/'
    root = '../data' # have to run in code/ folder
    # ==================== ADD YOUR ROOT HERE ==================== 
    #root = ''
    info = INFO[args.data]
    DataClass = getattr(medmnist, info['python_class'])
    test_dataset = DataClass(split='test', download=True, root=root)

    test_data = test_dataset.imgs 
    test_labels = test_dataset.labels[:, args.task_index]
    
    test_labels[test_labels != args.pos_class] = 99
    test_labels[test_labels == args.pos_class] = 1
    test_labels[test_labels == 99] = 0

    test_data = test_data/255.0
    test_data = torch.tensor(test_data, dtype=torch.float32)
    test_labels = torch.tensor(test_labels) 

    test_dataset = dataset(test_data, test_labels, trans=eval_transform)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.test_batchsize, shuffle=False, num_workers=0)

    if 1 != args.eval_only:
        train_dataset = DataClass(split='train', download=True, root=root)

        train_data = train_dataset.imgs 
        train_labels = train_dataset.labels[:, args.task_index]
    
        train_labels[train_labels != args.pos_class] = 99
        train_labels[train_labels == args.pos_class] = 1
        train_labels[train_labels == 99] = 0

        train_data = train_data/255.0
        train_data = torch.tensor(train_data, dtype=torch.float32)
        train_labels = torch.tensor(train_labels) 

        train_dataset = dataset(train_data, train_labels, trans=train_transform)

        #Undersampling Code
        #Count # of instances in each class
        class_counts = torch.bincount(train_labels)
        minClassCount = torch.min(class_counts).item()

        #Indices - each class
        classIndices = [torch.where(train_labels == i)[0] for i in range(2)]

        #Randomly sample from majority class
        usampIndices = torch.cat([indices[torch.randperm(len(indices))[:minClassCount]] for indices in classIndices])

        #New dataset with undersampled indices
        usamp_Data = train_data[usampIndices]
        usamLabels = train_labels[usampIndices]


        train_dataset = dataset(usamp_Data, usamLabels, trans=train_transform)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.train_batchsize, shuffle=True, num_workers=0)


        #Oversampling Code
        '''
        train_labels_binary = (train_labels == args.pos_class).to(torch.bool)
        class_counts = torch.bincount(train_labels)
        class_weights = 1. / class_counts.float()
        sample_weights = class_weights[train_labels_binary.to(torch.long)]
        
        sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)

        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.train_batchsize, shuffle=False, sampler=sampler, num_workers=0)
        '''



        #Comment out if Oversampling or Undersampling
        #train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.train_batchsize, shuffle=True, num_workers=0)

        '''
        #Visualize training data after processing
        inputs, classes = next(iter(train_loader))
        out = make_grid(inputs)
        imshow(out, title=[str(x.item()) for x in classes])
        plt.show()
        '''
        
 
    from libauc.models import resnet18 as ResNet18
    #from libauc.models import resnet50 as ResNet50
    from libauc.losses import AUCMLoss
    
    from torch.nn import BCELoss 
    from torch.optim import SGD
    from libauc.optimizers import PESG 
    net = ResNet18(pretrained=False) 
    #net = ResNet50(pretrained=False)
    net = net.to(device)  
    
    if args.loss == "CrossEntropy" or args.loss == "CE" or args.loss == "BCE":
        loss_fn = BCELoss() 
        optimizer = SGD(net.parameters(), lr=0.1)
    elif args.loss == "AUCM":
        loss_fn = AUCMLoss()
        optimizer = PESG(net.parameters(), loss_fn=loss_fn, lr=args.lr, margin=args.margin)
     
    if 1 != args.eval_only:
        train(net, train_loader, test_loader, loss_fn, optimizer, epochs=args.epochs)
    
    # to save a checkpoint in training: torch.save(net.state_dict(), "saved_model/test_model") 
    if 1 == args.eval_only: 
        net.load_state_dict(torch.load(args.saved_model_path)) 
        evaluate(net, test_loader) 

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
     
if __name__ == "__main__":
    main_worker()
