import torch
from torchvision.datasets import MNIST
from torchvision import transforms
from torch.utils.data import DataLoader
from torch import nn
import os
from PIL import Image
import numpy as np
from torchvision.datasets import MNIST
from torchvision import transforms
from torch.utils.data import DataLoader
from torch import nn
import os
import matplotlib.pyplot as plt

arccuracy_list=[]

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

train_data=MNIST(root='minist/data',train=True,transform=transform,download=True)
train_loader=DataLoader(train_data,batch_size=64,shuffle=True)
test_data=MNIST(root='minist/data',train=False,transform=transform,download=True)
test_loader=DataLoader(test_data,batch_size=64,shuffle=False)


class Model(nn.Module):
    def __init__(self):
        super(Model,self).__init__()
        self.fc1=nn.Linear(28*28,256)
        self.fc2=nn.Linear(256,64)
        self.fc3=nn.Linear(64,10)
    def forward(self,x):
        x=x.view(-1,784)
        x=torch.relu(self.fc1(x))
        x=torch.relu(self.fc2(x))
        x=torch.log_softmax(self.fc3(x),-1)
        return x
    
model=Model()
criteria=nn.CrossEntropyLoss()
optimizer=torch.optim.Adam(model.parameters(),lr=0.001)

if os.path.exists('minist/model/model.pkl'):
    model.load_state_dict(torch.load('minist/model/model.pkl'))

train_loss_list=[]
train_count_list=[]

def train(epoch):
    model.train()
    for index,data in enumerate(train_loader):
        inputs,labels=data
        optimizer.zero_grad()
        y_pred=model(inputs)
        loss=criteria(y_pred,labels)
        loss.backward()
        optimizer.step()
        if index%100==0:
            torch.save(model.state_dict(),'minist/model/model.pkl')
            torch.save(optimizer.state_dict(),'minist/model/optimizer.pkl')
            print('epoch:',epoch,'index:',index,'loss:',loss.item())

def test():
    model.eval()
    correct=0
    total=0
    for data in test_loader:
        inputs,labels=data
        y_pred=model(inputs)
        _,predicted=torch.max(y_pred.data,dim=1)
        total+=labels.size(0)
        correct+=(predicted==labels).sum().item()
    print('accuracy:',correct/total)
    arccuracy_list.append(correct/total)

if __name__=='__main__':
    for i in range(100):
        train(i)
        test()
    plt.plot(arccuracy_list)
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.show()