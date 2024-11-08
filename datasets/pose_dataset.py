import torch
from torch.utils.data import Dataset

class PoseDataset(Dataset):
    def __init__(self, data_2d, data_3d):
        self.data_2d = data_2d
        self.data_3d = data_3d

    def __len__(self):
        return len(self.data_2d)

    def __getitem__(self, idx):
        keypoints_2d = self.data_2d[idx]
        keypoints_3d = self.data_3d[idx]
        return torch.FloatTensor(keypoints_2d), torch.FloatTensor(keypoints_3d)
