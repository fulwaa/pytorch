import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torch.autograd import grad

x_train = torch.tensor([
    [-1.2,1.1,2],
    [-0.6,1.4,1.2],
    [-7,3,2.1],
    [0.2,0.4,0.1]
])

y_train = torch.tensor([1,0,1.1,0])

x_test = torch.tensor([
    [-1.1,2.3,1],
    [-2.1,1,3.2]
])

y_test = torch.tensor([0,1.1])


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
    

class LinearRegressionModel(torch.nn.Module):
    def __init__(self):
        super(LinearRegressionModel,self).__init__()
        self.linear = torch.nn.Linear(3,1) 

    def forward(self,x):
        y_pred = self.linear(x)
        return y_pred
    

train_ds = ToyDataset(x_train,y_train)
test_ds = ToyDataset(x_test,y_test)

torch.manual_seed(123)

train_loader = DataLoader(
    dataset= train_ds,
    batch_size=2,
    shuffle=True,
    num_workers=0
)

test_loader = DataLoader(
    dataset=test_ds,
    batch_size=2,
    shuffle = False,
    num_workers=0
    )

model = LinearRegressionModel()
criterion = torch.nn.MSELoss(size_average=False)
optimizer = torch.optim.SGD(model.parameters(),lr=0.01)

for epoch in range(20):
    for idx,(x,y) in enumerate(train_loader):

        pred_y = model(x)
        loss = criterion(pred_y,y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print('epoch{},loss{}'.format(epoch,loss.item()))

new_var = torch.Tensor([[4.0,3,1]])
pred_y = model(new_var)
print("predict (after training)", model(new_var).item())
