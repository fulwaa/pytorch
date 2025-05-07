import torch

class NeuralNetwork(torch.nn.Module):
    def __init__(self,num_input,num_outputs):
        super().__init__()#inherit the nn module
        self.layers = torch.nn.Sequential(
            torch.nn.Linear(num_input,30),
            torch.nn.ReLU(),

            torch.nn.Linear(30,20),
            torch.nn.ReLU(),
            torch.nn.Linear(20,num_outputs),

        )
    def forward(self,x):
        logits = self.layers(x)
        return logits

model = NeuralNetwork(10,2)
print(model)