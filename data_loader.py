import torch
from torch.utils.data import Dataset

x_train = torch.tensor([

    [-1.2,3.1],
    [-0.9,2.9],
    [-0.5,2.6],
    [2.3,-1.1],
    [2.7,-1.5]
])

y_train = torch.tensor([0,0,0,1,1])

x_test = torch.tensor([
    [-0.8,2.8],
    [2.6,-1.6]
])

y_test = torch.tensor([0,1])

class ToyDataset(Dataset):
    def __init__(self,x,y):
        self.features = x
        self.labels = y 

    def __getitem__(self, index):
        one_x = self.features[index]
        one_y = self.labels[index]
        return one_x,one_y

    def __len__(self):
        return self.labels.shape[0]

train_ds = ToyDataset(x_train,y_train)
test_ds = ToyDataset(x_test,y_test)
for i in train_ds:
    print(i)


    