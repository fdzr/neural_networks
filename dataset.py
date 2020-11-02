from torch.utils.data import Dataset
import torch


t_cg = torch.manual_seed(1547)


class RandomDataSet(Dataset):
    def __init__(self, N, F, C):
        self.X = torch.rand(N, F)
        tmp = torch.eye(C)
        self.Y = tmp[torch.randint(0, C, (N,))]
        self.num_features = F

    def __len__(self):
        return self.X.size()[0]

    def __getitem__(self, i):
        return self.X[i], self.Y[i]
