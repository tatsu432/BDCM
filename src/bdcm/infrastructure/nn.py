import torch


class MakeDataset(torch.utils.data.Dataset):
    def __init__(self, x, y=None):
        self.x = x
        self.y = y

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, i):
        x = torch.from_numpy(self.x[i]).float()
        if self.y is not None:
            y = torch.from_numpy(self.y[i]).float()
            return x, y
        return x


class Net_x(torch.nn.Module):
    """MLP for diffusion decoder (same architecture as original nn.Net_x)."""

    def __init__(self, num_input: int):
        super().__init__()
        self.fc1 = torch.nn.Linear(num_input, 128)
        self.fc2 = torch.nn.Linear(128, 256)
        self.fc3 = torch.nn.Linear(256, 256)
        self.fc4 = torch.nn.Linear(256, 1)

    def forward(self, x):
        x = torch.nn.functional.silu(self.fc1(x))
        x = torch.nn.functional.silu(self.fc2(x))
        x = torch.nn.functional.silu(self.fc3(x))
        return self.fc4(x)
