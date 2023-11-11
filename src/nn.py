import torch


# Define the data class to do mini batch learning
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

        if self.y is not None:
            return x, y
        else:
            return x

"""Neural Network: 3 hidden layers
- 1st layer: 128 nodes
- 2nd layer: 256 nodes
- 3rd layer: 256 nodes
"""

# # Define the nueral network to be used for DCM and BDCM
# Input: the number of input size
# Output: the object of neural network
class Net_x(torch.nn.Module):

    def __init__(self, num_input):
        super(Net_x, self).__init__()
        self.fc1 = torch.nn.Linear(num_input, 128)
        self.fc2 = torch.nn.Linear(128, 256)
        self.fc3 = torch.nn.Linear(256, 256)
        self.fc4 = torch.nn.Linear(256, 1)

    def forward(self, x):
        x = torch.nn.functional.silu(self.fc1(x))
        x = torch.nn.functional.silu(self.fc2(x))
        x = torch.nn.functional.silu(self.fc3(x))
        x = self.fc4(x)
        return x
    