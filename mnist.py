import torch
import torchvision
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torch.autograd import grad

img_size = 16
composed = transforms.Compose([transforms.Resize((img_size,img_size)),transforms.ToTensor()])

mnist_trainset = datasets.MNIST(root='./data',train=True,download=True,transform=composed)
mnist_testset = datasets.MNIST(root='./data',train=False,download=True,transform=composed)
print(len(mnist_trainset))
print(len(mnist_testset))

train_loader = DataLoader(
    dataset=mnist_trainset,
    batch_size= 2,
    shuffle = True,
    num_workers= 0

)

test_loader = DataLoader(
    dataset=mnist_testset,
    batch_size=2,
    shuffle=False,
    num_workers=0
)

class NeuralNetwork(torch.nn.Module):
    def __init__(self, num_input,num_output):
        super().__init__()
        self.layers = torch.nn.Sequential(
            torch.nn.Linear(num_input,30),
            torch.nn.ReLU(),

            torch.nn.Linear(30,20),
            torch.nn.ReLU(),
            torch.nn.Linear(20,num_output)
        )
    
    def forward(self,x):
        logits = self.layers(x)
        return logits
    
model = NeuralNetwork(16*16,10)
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(),lr=0.01)

for epoch in range(20):
    model.train()
    for idx,(x,y) in enumerate(train_loader):
        x = x.view(x.size(0), -1)  # Flatten the image
        pred_y = model(x)
        loss = criterion(pred_y,y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print('epoch{},loss{}'.format(epoch,loss.item()))

    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for feat,lab in test_loader:
            feat = feat.view(feat.size(0),-1)
            outputs = model(feat)
            _,predicted = torch.max(outputs.data,1)
            total += lab.size(0)
            correct += (predicted == lab).sum().item()

    print(f"Epoch {epoch} - Test Accuracy: {100 * correct / total:.2f}%")

torch.save(model.state_dict(), "model.pth")