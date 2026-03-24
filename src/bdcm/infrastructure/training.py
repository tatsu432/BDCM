from __future__ import annotations

from typing import TYPE_CHECKING, List, Sequence

import matplotlib.pyplot as plt
import numpy as np
import torch

from bdcm.infrastructure.nn import MakeDataset, Net_x

if TYPE_CHECKING:
    from bdcm.config import ExperimentConfig


def train_neural_nets(
    array_input_x: Sequence[np.ndarray],
    epsilon_for_x: np.ndarray,
    array_index_for_epsilon: np.ndarray,
    array_num_input_for_nn: np.ndarray,
    array_titles: np.ndarray,
    config: ExperimentConfig,
) -> List[torch.nn.Module]:
    """Train one Net_x per head; optionally plot training loss."""
    nets: List[torch.nn.Module] = []
    epoch_losses: List[List[float]] = []

    for i in range(len(array_input_x)):
        idx = int(array_index_for_epsilon[i])
        dataset = MakeDataset(
            array_input_x[i],
            epsilon_for_x[idx].reshape(-1, 1),
        )
        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=config.batch_size, shuffle=True
        )
        net = Net_x(int(array_num_input_for_nn[i]))
        net.train()
        optimizer = torch.optim.Adam(net.parameters(), lr=config.learning_rate)
        criterion = torch.nn.MSELoss()
        losses: List[float] = []
        for _ in range(config.num_epochs):
            running_loss = 0.0
            for inputs, labels in dataloader:
                outputs = net(inputs)
                loss = criterion(outputs, labels)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                running_loss += float(loss.detach().cpu().numpy())
            losses.append(running_loss)
        nets.append(net)
        epoch_losses.append(losses)

    if config.flags.plot_nn_train:
        for i, losses in enumerate(epoch_losses):
            fig = plt.figure()
            ax = fig.add_subplot()
            ax.plot(list(range(len(losses))), losses)
            ax.set_xlabel("number of epochs")
            ax.set_ylabel("loss")
            ax.set_yscale("log")
            ax.set_title("${}$".format(array_titles[i]))
            fig.show()

    return nets
