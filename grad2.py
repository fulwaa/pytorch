import torch
import torch.nn.functional as F
from torch.autograd import grad

y = torch.tensor([1.0])
x1 = torch.tensor([1.1])
w1 = torch.tensor([2.2], requires_grad = True)
b = torch.tensor([0.0], requires_grad= True)

z = x1*w1+b

a = torch.sigmoid(z)

loss = F.binary_cross_entropy(a,y)
loss.backward()
print(w1.grad)
print(b.grad)