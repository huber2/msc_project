from os.path import dirname, abspath
import numpy as np
from matplotlib import pyplot as plt
import torch
from torch.utils.data import Dataset, DataLoader
from preprocessing import DemoPreprocessor
#from model_conv1 import ConvNet1
#from mlp import MLP

DIR_PATH = dirname(abspath(__file__)) + '/../'
demos_path = DIR_PATH + 'data/demo_reach_object_22_07_07_10_51_44.npz'

#model = ConvNet1(n_classes=6)
#model = MLP(layers_dim=(64*64*3, 512, 6))
model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=False, num_classes=7)
print('MODEL INFO:', model, sep='\n')
print("Number of learnable parameters:", sum(np.prod(np.array(p.shape) for p in model.parameters())))


preprocessor = DemoPreprocessor()
preprocessor.load_trajectories(demos_path)
preprocessor.dataset_split(n_test=10, n_valid=10, n_train=100)
preprocessor.normalization()
preprocessor.create_datasets()

train_loader = DataLoader(preprocessor.train_set, batch_size=64, shuffle=True, drop_last=True)
v_loader = DataLoader(preprocessor.v_set, batch_size=64, shuffle=False)


def train(model, train_loader, v_loader, n_epochs):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    
    loss_logger = {'loss':[], 'vloss':[], 'loss_class':[], 'vloss_class':[]}

    for epoch in range(n_epochs):
        print(f"Starting epoch {epoch}/{n_epochs}")
        model.train()
        for batch, (x, y) in enumerate(train_loader):
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            y_pred = model.forward(x)
            loss = criterion(y_pred, y)
            loss.backward()
            optimizer.step()
            if batch % 100 == 0:
                model.eval()
                with torch.no_grad():
                    print(f"batch {batch}/{len(train_loader)}; loss={loss.item():>6f}")
                    loss_logger['loss'].append(loss.item())
                    loss_per_class = ((y - y_pred)**2).mean(axis=0)
                    loss_logger['loss_class'].append(loss_per_class)
                    x_val, y_val = v_loader.dataset.x.to(device), v_loader.dataset.y.to(device)
                    y_pred = model.forward(x_val)
                    v_loss_per_class = ((y_val - y_pred)**2).mean(axis=0)
                    v_loss = v_loss_per_class.mean()
                    loss_logger['vloss'].append(v_loss)
                    loss_logger['vloss_class'].append(v_loss_per_class)
                    print(f"Epoch {epoch}/{n_epochs}; batch {batch}; v_loss={v_loss:>6f}")
                model.train()

    return loss_logger