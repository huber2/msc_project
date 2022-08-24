from os.path import dirname, abspath
import numpy as np
from matplotlib import pyplot as plt
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from model_very_simple_conv_32 import ConvNet


DIR_PATH = dirname(abspath(__file__)) + '/../'

# Check data

colors = (
    'FF0000',
    '00FF00',
    '0000FF',
    )


class DemoDataset(Dataset):
    def __init__(self, data_x, data_y):
        self.x = torch.as_tensor(data_x, dtype=torch.float32).permute(0, 3, 1, 2)
        self.y = torch.as_tensor(data_y, dtype=torch.float32)
    def __len__(self):
        return len(self.y)
    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


def normalize(x_data, y_data, max_speed_linear=0.1, max_speed_angular=0.2):
    """"Normalize input images with values between 0 and 1,
    and outputs with values between -1 and 1.
    """
    normal_x = x_data / 255
    n_outputs = y_data.shape[1]
    if n_outputs == 3:
        normal_y = y_data / max_speed_linear
    else:
        normal_y = y_data / np.array([3*[max_speed_linear] + 3*[max_speed_angular]])
    return normal_x, normal_y



def mask_images(imgs, img_size=32, mask_size_range=(3, 20)):
    """Augment image data by partially masking each image with a squared mask of random color,
    random size and placed at a random position in the image

    Args:
        imgs (array): images to be augmented
        img_size (int): size of the images (assumed square)
        mask_size_range: (minimum, maximum) size that the mask can take

    """
    new_imgs = imgs.copy()
    for im in new_imgs:
        mask_size = np.random.randint(mask_size_range[0], mask_size_range[1])
        mask_pos = np.random.randint(0, img_size-mask_size+1, 2)
        color = np.random.randint(0, 256, 3)
        im[mask_pos[0]:mask_pos[0]+mask_size, mask_pos[1]:mask_pos[1]+mask_size] = color
    return new_imgs


def prepare_data(color_hex, n_trajectories, train_prop, n_augmented):
    demo_data = np.load(f'{DIR_PATH}data/demo_reach_cuboid_{color_hex}_camera32.npz')
    assert n_trajectories < len(demo_data['step_marker']), "Not enough trajectories"

    # Seperate training and validation set
    n_train = int(train_prop * n_trajectories)
    n_val = n_trajectories - n_train

    print(f"n_train={n_train}, n_val={n_val}")

    idx_train = demo_data['step_marker'][n_train]
    idx_val = demo_data['step_marker'][n_trajectories]

    train_imgs = demo_data['demo_image_sequences'][:idx_train]
    train_act = demo_data['demo_action_sequences'][:idx_train]
    val_imgs = demo_data['demo_image_sequences'][idx_train:idx_val]
    val_acts = demo_data['demo_action_sequences'][idx_train:idx_val]
    train_imgs.shape, train_act.shape, val_imgs.shape, val_acts.shape

    # Data augmentation if n_augmented > 0
    masked_train_images = [mask_images(train_imgs) for _ in range(n_augmented)]
    train_x = np.concatenate([train_imgs] + masked_train_images)
    train_y = np.concatenate([train_act] * (1 + n_augmented))


    masked_val_images = [mask_images(val_imgs) for _ in range(n_augmented)]
    val_x = np.concatenate([val_imgs] + masked_val_images)
    val_y = np.concatenate([val_acts] * (1 + n_augmented))

    # Normalize data
    normal_train_x, normal_train_y = normalize(train_x, train_y)
    normal_val_x, normal_val_y = normalize(val_x, val_y)

    train_set = DemoDataset(normal_train_x, normal_train_y)
    v_set = DemoDataset(normal_val_x, normal_val_y)

    return train_set, v_set


# Check datasets

def check(train_set, v_set):
    print(train_set.x.shape, train_set.y.shape, v_set.x.shape, v_set.y.shape, sep='\n')

    for i in range(3):
        print(train_set.x[:, i].amin(), train_set.x[:, i].amax())
        
    for i in range(3):
        print(train_set.y[:, i].amin(), train_set.y[:, i].amax())


    n_display_examples, n_rows, n_cols = 24, 4, 6
    ids = np.random.choice(len(train_set), n_display_examples, replace=False)

    plt.figure()
    for i, id in enumerate(ids):
        plt.subplot(n_rows, n_cols, i+1)
        plt.imshow(train_set.x[id].permute(1, 2, 0))
        plt.axis('off')
    plt.show()


def train(model, train_set, v_set, lr, bs, stop_criterion, max_epochs, model_name='current_model', device_name='cuda:0'):
    train_loader = DataLoader(train_set, batch_size=bs, shuffle=True, drop_last=True)
    device = torch.device(device_name)
    model.to(device)
    vx, vy = v_set.x.to(device), v_set.y.to(device)
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    loss_logger = {'loss':[], 'vloss':[], 'loss_class':[], 'vloss_class':[]}
    best_v_loss = np.inf
    stop_counter = 0

    for epoch in range(max_epochs):
        with tqdm(train_loader, unit='batch') as tepoch:
            tepoch.set_description(f'Epoch {epoch}/{max_epochs} max ; best_vloss={best_v_loss:.6f}')
            for (x, y) in tepoch:
                model.train()
                x, y = x.to(device), y.to(device)
                optimizer.zero_grad()
                y_pred = model.forward(x)
                loss = criterion(y_pred, y)
                loss.backward()
                optimizer.step()

                tepoch.set_postfix_str(f'loss={loss.item():.6f}', refresh=False)

            model.eval()
            with torch.no_grad():
                loss_logger['loss'].append(loss.item())
                loss_per_class = ((y - y_pred)**2).mean(axis=0).cpu().numpy()
                loss_logger['loss_class'].append(loss_per_class)
                vy_pred = model.forward(vx)
                v_loss_per_class = ((vy - vy_pred)**2).mean(axis=0).cpu().numpy()
                v_loss = v_loss_per_class.mean().item()
                loss_logger['vloss'].append(v_loss)
                loss_logger['vloss_class'].append(v_loss_per_class)
                if v_loss < best_v_loss:
                    best_v_loss = v_loss
                    stop_counter = 0
                    torch.save(model.state_dict(), f'{DIR_PATH}/data/{model_name}.pth')
                else:
                    stop_counter += 1
                    if stop_counter > stop_criterion:
                        print('Early stopping')
                        break

    return loss_logger


def plot_loss(loss_log):
    plt.figure()
    plt.plot(np.stack(loss_log['vloss_class']), '.-')
    plt.yscale('log')
    plt.title("Validation loss per output on predicted linear velocity (along x,y,z)", fontsize=13)
    plt.ylabel("Mean Square Error loss - (normalized)", fontsize=12)
    plt.xlabel("epoch", fontsize=12)
    plt.legend(('dx', 'dy', 'dz'))
    plt.grid()
    plt.show()

    plt.figure()
    plt.plot(loss_log['loss'], '.-', label='Training loss')
    plt.plot(loss_log['vloss'], '.-', label='Validation loss')
    plt.title("MSE Loss on predicted linear velocity (along x,y,z)", fontsize=13)
    plt.ylabel("Mean Square Error loss - (normalized)", fontsize=12)
    plt.xlabel("epoch", fontsize=12)
    plt.yscale('log')
    plt.legend()
    plt.grid()
    plt.show()


def save_loss(loss_log, name):
    for k,v in loss_log.items():
        loss_log[k] = np.array(v)
    np.savez_compressed(name, **loss_log)

def plot_loss(loss_log):
    plt.figure()
    plt.plot(np.stack(loss_log['vloss_class']), '.-')
    plt.yscale('log')
    plt.title("Validation loss per output on predicted linear velocity (along x,y,z)", fontsize=13)
    plt.ylabel("Mean Square Error loss - (normalized)", fontsize=12)
    plt.xlabel("epoch", fontsize=12)
    plt.legend(('dx', 'dy', 'dz'))
    plt.grid()
    plt.show()

    plt.figure()
    plt.plot(loss_log['loss'], '.-', label='Training loss')
    plt.plot(loss_log['vloss'], '.-', label='Validation loss')
    plt.title("MSE Loss on predicted linear velocity (along x,y,z)", fontsize=13)
    plt.ylabel("Mean Square Error loss - (normalized)", fontsize=12)
    plt.xlabel("epoch", fontsize=12)
    plt.yscale('log')
    plt.legend()
    plt.grid()
    plt.show()


def main():
    dataset_n_trajectories = [50, 100, 150, 250]
    colors_trained = colors[:] # 0:Red
    
    for col in colors_trained:
        for dataset_size in dataset_n_trajectories:
            train_set, v_set = prepare_data(col, dataset_size, 0.8, n_augmented=0)
            check(train_set, v_set)
            model = ConvNet(n_classes=3)
            loss_log = train(model, train_set, v_set, lr=0.0005, bs=512, stop_criterion=20, max_epochs=500, model_name=f'nn32_no_aug_model_{col}_ntraj_{dataset_size}', device_name='cpu')
            save_loss(loss_log, name=f'{DIR_PATH}/data/loss_nn32_no_aug_model_{col}_ntraj_{dataset_size}')
            plot_loss(loss_log)



if __name__ == '__main__':
    main()