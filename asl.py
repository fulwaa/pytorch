import torch
from torch.utils.data import Dataset, random_split
import torchvision 
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch.nn as nn

transform = transforms.Compose([transforms.Resize((64,64)),transforms.ToTensor()])

# Load the full dataset
dataset = datasets.ImageFolder(root='data/asl_alphabet/asl_alphabet_train', transform=transform)

# Split into train and validation sets
train_size = int(0.8 * len(dataset))
valid_size = len(dataset) - train_size
train_dataset, valid_dataset = random_split(dataset, [train_size, valid_size])

train_loader = DataLoader(
    dataset= train_dataset,
    batch_size=2,
    shuffle= True,
    num_workers=0
)

valid_loader = DataLoader(
    dataset=valid_dataset,
    batch_size= 2,
    shuffle= True,
    num_workers=0
)

class cnn(nn.Module):
    def __init__(self,out1,out2):
        super(cnn,self).__init__()

        self.cnn1 = nn.Conv2d(in_channels=3,out_channels=out1,kernel_size=5,stride=1,padding=2)
        self.max = nn.MaxPool2d(kernel_size=2)
        self.cnn2 = nn.Conv2d(in_channels=out1,out_channels=out2,kernel_size=5,stride=1,padding=2)
        self.fc1 = nn.Linear(out2 * 16 * 16, 29)  # instead of 4*4


    


    def forward(self,x):
        x= self.cnn1(x)
        x= self.max(x)
        x= self.cnn2(x)
        x= self.max(x)
        x= x.view(x.size(0),-1)
        x= self.fc1(x)
        return x
    
model = cnn(out1=64,out2=32)
criterion = nn.CrossEntropyLoss()
learning_rate = 0.01
optimizer = torch.optim.SGD(model.parameters(),lr=learning_rate)

# print(model)

for epoch in range(10):
    model.train()
    for idx,(features,labels) in enumerate(train_loader):
        logits = model(features)
        loss =  criterion(logits,labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f'{epoch}:{loss}')
        