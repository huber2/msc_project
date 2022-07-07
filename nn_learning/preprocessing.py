import numpy as np
import torch
from torch.utils.data import Dataset


class DemoDataset(Dataset):
    def __init__(self, data_x, data_y):
        self.x = torch.as_tensor(data_x, dtype=torch.float32).permute(0, 3, 1, 2)
        self.y = torch.as_tensor(data_y, dtype=torch.float32)
    def __len__(self):
        return len(self.y)
    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


class DemoPreprocessor:
    def __init__(self):
        self.n_all = 0

    def load_trajectories(self, demos_path):
        print("Loading data from:", demos_path)
        demos = np.load(demos_path, allow_pickle=True)
        self.image_seqs = demos['demo_image_sequences']
        self.action_seqs = demos['demo_action_sequences']
        self.target_pos_seqs = demos['demo_target_poses_sequences']
        self.n_traj = len(self.action_seqs)
        print("Done. Number of trajectories:", self.n_traj)

    def dataset_split(self, n_test, n_valid, n_train):
        print(f"Creating train, validation and test sets with respectively {n_train}, {n_valid} and {n_test} tajectories")
        assert self.n_traj >= n_valid + n_train + n_test
        random_idx = np.random.permutation(self.n_traj)

        test_idx = random_idx[: n_test]
        val_idx = random_idx[n_test : n_test+n_valid]
        train_idx = random_idx[n_test+n_valid : n_test+n_valid+n_train]

        test_img_seqs = self.image_seqs[test_idx]
        test_act_seqs = self.action_seqs[test_idx]
        test_targ_seqs = self.target_pos_seqs[test_idx]

        val_img_seqs = self.image_seqs[val_idx]
        val_act_seqs = self.action_seqs[val_idx]
        val_targ_seqs = self.target_pos_seqs[val_idx]

        train_img_seqs = self.image_seqs[train_idx]
        train_act_seqs = self.action_seqs[train_idx]
        train_targ_seqs = self.target_pos_seqs[train_idx]

        data_seqs = (
            test_img_seqs, val_img_seqs, train_img_seqs, 
            test_act_seqs, val_act_seqs, train_act_seqs,
            test_targ_seqs, val_targ_seqs, train_targ_seqs)

        data_concat = tuple(map(np.concatenate, data_seqs))
        self.test_x, self.val_x, self.train_x = data_concat[:3]
        self.test_y = np.concatenate([data_concat[3], data_concat[6]], axis=1)
        self.val_y = np.concatenate([data_concat[4], data_concat[7]], axis=1)
        self.train_y = np.concatenate([data_concat[5], data_concat[8]], axis=1)

    def update_train_output_mean_std(self):
        self.mean = self.train_y.mean(axis=0).reshape(1, -1)
        self.std = self.train_y.std(axis=0).reshape(1, -1)

    def standard_normalize(self, x, mean, std):
        return (x - mean) / std

    def normalization(self):
        self.normal_test_x = self.test_x
        self.normal_val_x = self.val_x
        self.normal_train_x = self.train_x

        self.update_train_output_mean_std()

        self.normal_test_y = self.standard_normalize(self.test_y, self.mean, self.std)
        self.normal_val_y = self.standard_normalize(self.val_y, self.mean, self.std)
        self.normal_train_y = self.standard_normalize(self.train_y, self.mean, self.std)

    def create_datasets(self):
        self.train_set = DemoDataset(self.normal_train_x, self.normal_train_y)
        self.v_set = DemoDataset(self.normal_val_x, self.normal_val_y)
        self.test_set = DemoDataset(self.normal_test_x, self.normal_test_y)
        print('train_set: dim in:', self.train_set.x.shape,'out:', self.train_set.y.shape)
        print('v_set dim: in:', self.v_set.x.shape,'out:', self.v_set.y.shape)
        print('test_set dim: in:', self.test_set.x.shape,'out:', self.test_set.y.shape)