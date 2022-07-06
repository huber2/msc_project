import torch
import torch.nn as nn

class MLP(nn.Module):   
    def __init__(self, layers_dim):
        super().__init__()
        layers = []
        for i in range(len(layers_dim)-1):
            layers.append(nn.Linear(layers_dim[i], layers_dim[i+1]))
            if i != len(layers_dim)-2:
                layers.append(nn.ReLU(inplace=True))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        x = torch.flatten(x, 1)
        x = self.layers(x)
        return x