import os
import numpy as np
from matplotlib import pyplot as plt
from model_conv1 import ConvNet1
import torch


DIR_PATH = os.getcwd() + '/'

model = ConvNet1(n_classes=7)

data_traj = np.load(DIR_PATH + 'data/demos_reach.npz', allow_pickle=True)

x = data_traj['demo_image_sequences']
y = data_traj['demo_action_sequences']

n_examples = len(y)

x_test = x[:100]
y_test = y[:100]

x_validation = x[100:200]
y_validation = y[100:200]

x_train = x[200+700:]
y_train = y[22+n_validation:]



imgs_all = np.concatenate(data['arr_0'].all()['images'])
inp = torch.tensor(imgs_all.transpose(0, 3, 1, 2), dtype=torch.float32) / 255
dataset = (inp, label)

'''
n_examples = 10_000
small_dataset = (dataset[0][:n_examples], dataset[1][:n_examples])
small_dataset[0].shape, small_dataset[1].shape
'''

action_dim = dataset[1].shape[1]
model.fc = torch.nn.Linear(512, action_dim)

print('MODEL INFO:')
print(model)

print('\n\nget_device_properties(0):\n', torch.cuda.get_device_properties(0))

# Main entry point
def train(model, dataset, num_epochs, batch_size, valid_prop=0, early_stopping=False):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    r = torch.cuda.memory_reserved(0)
    a = torch.cuda.memory_allocated(0)
    info = np.array([r, a, r-a])/10e6
    print('gpu memory after adding model:', info)
    
    print('device', device)
    model.to(device)
    inputs, labels = dataset[0], dataset[1]

    r = torch.cuda.memory_reserved(0)
    a = torch.cuda.memory_allocated(0)
    info = np.array([r, a, r-a])/10e6
    print('mem', info)
    
    # Prepare data
    num_examples = len(inputs)
    print("num_examples", num_examples)
    
    num_validation = int(valid_prop * num_examples)
    num_train = num_examples - num_validation
    num_batches = int(np.ceil(num_epochs * num_train / batch_size))
    print("num_batches", num_batches)

    permutation = np.random.permutation(num_examples)
    train_idx = permutation[:num_train]
    validation_idx = permutation[num_train:]
    
    train_data = inputs[train_idx]
    train_labels = labels[train_idx]
    validation_data = inputs[validation_idx].to(device)
    validation_labels = labels[validation_idx].to(device)
    
    print("x", train_data.shape)
    print("y", train_labels.shape)
    print('validation_data.shape', validation_data.shape)
    print('validation_labels.shape', validation_labels.shape)
    
    r = torch.cuda.memory_reserved(0)
    a = torch.cuda.memory_allocated(0)
    info = np.array([r, a, r-a])/10e6
    print('gpu memory after adding validation data:', info)
    
    
    permutations = np.concatenate([np.random.permutation(num_train) for _ in range(num_epochs+1)])
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    criterion = torch.nn.MSELoss()
    losses = []
    best_train_loss = np.inf

    if len(validation_data) > 0:
        with torch.no_grad():
            validation_pred = model.forward(validation_data)
            validation_loss = criterion(validation_pred, validation_labels)
            validation_losses = [validation_loss.item()]
            best_loss = validation_loss
    
    if early_stopping:
        early_stop_counter = 0
        early_stop_lim = 100
        t_start = time.time()
        t_lim = 600
    
    for i in range(num_batches):
        indices = permutations[i*batch_size:(i+1)*batch_size]
        
        x = train_data[indices].to(device)
        y = train_labels[indices].to(device)
        
        model.train()
        optimizer.zero_grad()
        y_pred = model.forward(x)
        loss = criterion(y_pred, y)
        loss.backward()
        optimizer.step()
        
        losses.append(loss.item())
        
        model.eval()
        if len(validation_data) > 0:
            with torch.no_grad():
                validation_pred = model.forward(validation_data)
                validation_loss = criterion(validation_pred, validation_labels)
                validation_losses.append(validation_loss.item())
                
                if validation_loss.item() < best_loss:
                    best_loss = validation_loss.item()
                    torch.save(model.state_dict(), PATH)
                    if early_stopping:
                        early_stop_counter = 0
                elif loss.item() >= validation_loss.item():
                    early_stop_counter = 0
                if early_stopping and (1.5 * loss.item() < validation_loss.item() or (validation_losses[-3] < validation_losses[-2]) and (validation_losses[-2] < validation_losses[-1])):
                    early_stop_counter += 1
                
        elif loss.item() < best_train_loss:
            best_train_loss = loss.item()
            torch.save(model.state_dict(), PATH)
        
        if i%(1+num_batches//100) == 0:
            info = f'Iteration {i}/{num_batches}'
            info += f', Epoch {(i*num_epochs)//num_batches}/{num_epochs}'
            info += f', Loss = {loss.item()}'
            if len(validation_data) > 0:
                info += f', Validation = {validation_loss.item()}'
            print(info)
        
        if early_stopping and (early_stop_counter > early_stop_lim or time.time() - t_start > t_lim):
            break
        
    torch.save(model.state_dict(), PATH2)
    model.load_state_dict(torch.load(PATH))
    model.eval()

    # Plot and save loss evolution
    fig, ax = plt.subplots()
    ax.set(xlabel='Iteration', ylabel='Loss', title='Loss Curve for Torch Example')
    ax.plot(range(len(losses)), losses, color='blue', label='Training loss')
    if len(validation_data) > 0:
        ax.plot(range(len(validation_losses)), validation_losses, color='r', label='Validation loss')
    plt.yscale('log')
    plt.legend()
    plt.grid()
    fig.savefig(PATH3)
    plt.show()
    
    return losses

if __name__ == '__main__':
    PATH = DIR_PATH + 'data/resnet18_best_bs128.pt'
    PATH2 = DIR_PATH + 'data/resnet18_final_itr_bs128.pt'
    PATH3 = DIR_PATH + 'data/loss_vs_iterations_resnet18_bs128.png'
    torch.cuda.empty_cache()
    train(model, dataset, num_epochs=20, batch_size=64, valid_prop=0.05)

