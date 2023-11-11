import numpy as np
import torch
import random
import matplotlib.pyplot as plt
import conf
from nn import MakeDataset, Net_x

array_title = ["DCM", "BDCM"]

# Define the function to normalize
def normalize(x):
    return (x - np.mean(x)) / np.std(x)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(s):
  # Set random seed for random
  random.seed(s)
  # Set random seed for NumPy
  np.random.seed(s)
  # Set random seed for PyTorch
  torch.manual_seed(s)
  if torch.cuda.is_available():
      torch.cuda.manual_seed_all(s)
  torch.backends.cudnn.benchmark = False
  torch.backends.cudnn.deterministic = True

def create_alpha_t_train_for_x(d, t_for_x):
  # Get the alpha_t for training
  # Initialize the alpha_t
  alpha_t_train_for_x = np.zeros((d, conf.n_obs))
  for i in range(conf.n_obs):
    for j in range(d):
      alpha_t_train_for_x[j][i] = conf.alpha_t[t_for_x[j][i] - 1]
  return alpha_t_train_for_x

# Define the function to create the first input to the neural network
def create_input_1(alpha_t, x, epsilon):
    return np.sqrt(alpha_t) * x + np.sqrt(1 - alpha_t) * epsilon



# Define the decoder
# Dec_i(Z_i, X_{pa_i})
# Input: exogenous node, array of parent nodes, and the index of the order NN was created
# Output: the last node in the reverse diffusion process
def DEC(x_parents, index_order, array_net_x):

    # Define the vector that preserves the variables obtained via the reverse diffusion process
    x_hat = np.zeros(conf.T + 1)
    # Initialize the start node of the reverse diffusion process
    x_hat[conf.T] = np.random.normal(0, 1)
    # Save the variables in the reverse diffusion process
    for i in range(1, conf.T + 1):
        t = conf.T + 1 - i

        # Change the network mode to evaluation
        array_net_x[index_order].eval()
        # Define the inputs
        # input_x = np.column_stack((np.array([x_hat[t]]), x_parents.T, np.array([t])))
        input_x = np.append(np.array([x_hat[t]]), x_parents)
        input_x = np.append(input_x, np.array([t]))
        input_x = np.array([input_x])
        # convert the input to tensor
        input_x_tensor = torch.from_numpy(input_x).float()
        # Make prediction
        with torch.no_grad():
            output_x_tensor = array_net_x[index_order](input_x_tensor)
        # Convert the output to numpy data
        output_x = output_x_tensor.data.numpy()
        # Calculate the next variable in the reverse diffusion process by the formula
        x_hat[t - 1] = np.sqrt(conf.alpha_t[t - 2] / conf.alpha_t[t - 1]) * x_hat[t] - output_x[0][0] * (np.sqrt(conf.alpha_t[t - 2] * (1 - conf.alpha_t[t - 1]) / conf.alpha_t[t - 1]) - np.sqrt(1 - conf.alpha_t[t - 2]))
    # Return the last variable in the reverse diffusion process
    return x_hat[0]





def train_and_plot_neural_net(array_input_x, epsilon_for_x, array_index_for_epsilon, array_num_input_for_nn, array_titles):
  """Train the Neural Network"""

  # Train the neural network that will be used for the decoding process
  # Initialize the array that save the trained nets and the losses
  array_net_x = np.array([])
  array_epoch_loss = np.array([])

  # loop for the neural net
  for i in range(len(array_input_x)):
    # Prepare dataset
    dataset = MakeDataset(array_input_x[i], epsilon_for_x[array_index_for_epsilon[i]].reshape(-1, 1))
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=conf.batch_size, shuffle=True)

    # Prepare model and training parameters
    # Instantiate the Neural Network
    net = Net_x(array_num_input_for_nn[i])
    # Chnage the Neural Network to the training mode
    net.train()
    # Define the optimizer
    optimizer = torch.optim.Adam(net.parameters(), lr=conf.learning_rate)
    # Define the criterion
    criterion = torch.nn.MSELoss()

    # Training
    epoch_loss = []
    for epoch in range(conf.num_epochs):
      # use 'dataloader' to start batch learning
      running_loss = 0   # loss in this epoch
      for inputs, labels in dataloader:
          outputs = net(inputs)
          loss = criterion(outputs, labels)
          optimizer.zero_grad()
          loss.backward()
          optimizer.step()
          # add loss of this batch to loss of epoch
          running_loss += loss.data.numpy().tolist()

      epoch_loss.append(running_loss)

    # Append the trained net and loss to the array
    if i == 0:
      array_net_x = net
      array_epoch_loss = epoch_loss
    else:
      array_net_x = np.append(array_net_x, net)
      array_epoch_loss = np.vstack((array_epoch_loss, epoch_loss))

  """Plot the loss of the training over the epoch"""

  num_neural_net = len(array_epoch_loss)

  # for loop over the neural network
  for i in range(num_neural_net):
    # Plot the loss over the epoch
    fig = plt.figure()
    ax = fig.add_subplot()
    ax.plot(list(range(len(array_epoch_loss[i]))), array_epoch_loss[i])
    ax.set_xlabel('number of epochs')
    ax.set_ylabel('loss')
    ax.set_yscale('log')
    ax.set_title('${}$'.format(array_titles[i]))
    fig.show()

  return array_net_x






def MMD(x, y, kernel):
    """Emprical maximum mean discrepancy. The lower the result
    the more evidence that distributions are the same.

    Args:
        x: first sample, distribution P
        y: second sample, distribution Q
        kernel: kernel type such as "multiscale" or "rbf"
    """
    xx, yy, zz = torch.mm(x, x.t()), torch.mm(y, y.t()), torch.mm(x, y.t())
    rx = (xx.diag().unsqueeze(0).expand_as(xx))
    ry = (yy.diag().unsqueeze(0).expand_as(yy))

    dxx = rx.t() + rx - 2. * xx # Used for A in (1)
    dyy = ry.t() + ry - 2. * yy # Used for B in (1)
    dxy = rx.t() + ry - 2. * zz # Used for C in (1)

    XX, YY, XY = (torch.zeros(xx.shape).to(device),
                torch.zeros(xx.shape).to(device),
                torch.zeros(xx.shape).to(device))

    if kernel == "multiscale":

        bandwidth_range = [0.2, 0.5, 0.9, 1.3]
        for a in bandwidth_range:
            XX += a**2 * (a**2 + dxx)**-1
            YY += a**2 * (a**2 + dyy)**-1
            XY += a**2 * (a**2 + dxy)**-1

    if kernel == "rbf":

        bandwidth_range = [10, 15, 20, 50]
        for a in bandwidth_range:
            XX += torch.exp(-0.5*dxx/a)
            YY += torch.exp(-0.5*dyy/a)
            XY += torch.exp(-0.5*dxy/a)

    return torch.mean(XX + YY - 2. * XY)







def create_array_array_MMD(array_interventions, array_array_DCM_BDCM_samples, true_sample, d, structural_eq, ind_cause, ind_result, array_u, array_array_MMD):
  # Calculate the number of intervention values
  num_interventions = np.size(array_interventions)

  # Show all the graphs and output MMD where we do(X_1 = x_1)
  figure, axis = plt.subplots(num_interventions, 2, figsize=(10, 4.5 * num_interventions))

  # Initialized the array that saves the values of MMD for DCM and BDCM
  array_MMD_DCM = np.array([])
  array_MMD_BDCM = np.array([])

  array_MMD_DCM_BDCM = np.zeros([2, num_interventions])

  # for loop for each intervened value
  for i in range(num_interventions):
    # Get the intervened value
    intervened_value_for_cause_node = array_interventions[i]
    # for loop for DCM or BDCM
    for j in range(2):
      # do(X_1 = x_1)
      # Plot samples from DCM or BDCM
      axis[i][j].hist(normalize(array_array_DCM_BDCM_samples[j][i]), 100, density = True, label = "sample")
      # Plot ground truth samples
      axis[i][j].hist(normalize(true_sample(d, structural_eq, ind_cause, ind_result, intervened_value_for_cause_node, array_u)), 100, density = True, alpha = 0.5, label = "target dist")
      axis[i][j].set_title("$X_{}|do(X_{} = {:.4})$ {}".format(ind_result + 1, ind_cause + 1, intervened_value_for_cause_node, array_title[j]))
      axis[i][j].legend()
      # Calculate MMD
      mmd_value = MMD(torch.tensor([normalize(array_array_DCM_BDCM_samples[j][i])]).T.to(device), torch.tensor([normalize(true_sample(d, structural_eq, ind_cause, ind_result, intervened_value_for_cause_node, array_u))[:conf.n_sample_DCM]]).T.to(device), "rbf")
      array_MMD_DCM_BDCM[j][i] = mmd_value.item()

  # Output the mean and standard deviation of MMD for DCM and BDCM
  # loop for DCM or BDCM
  for i in range(2):
      print("mean of MMD for {}: {:.3}".format(array_title[i], np.mean(array_MMD_DCM_BDCM[i])))
      print("standard deviation of MMD for {}: {:.3}".format(array_title[i], np.std(array_MMD_DCM_BDCM[i])))


  array_array_MMD.append(array_MMD_DCM_BDCM)



  



# Create the array that save the intervened value, samples from DCM, samples from BDCM
# Input: array of intervened values
# Output: DCM and BDCM samples
def save_array(array_interventions, sample_outcome_do_cause_DCM, sample_outcome_do_cause_BDCM, x, array_net_x):
  # Get the number of interventions
  num_intervention = np.size(array_interventions)

  # Initialize the array of samples for DCM and BDCM
  array_DCM_samples = np.array([])
  array_BDCM_samples = np.array([])

  # for each intervention
  for i in range(num_intervention):
    if np.any(array_DCM_samples) == False:
      # Plot the empirical distribution of DCM and true target
      array_DCM_samples = np.append(array_DCM_samples, np.array([sample_outcome_do_cause_DCM(array_interventions[i], x, array_net_x)]))
      # Plot the empirical distribution of BDCM and true target
      array_BDCM_samples = np.append(array_BDCM_samples, np.array([sample_outcome_do_cause_BDCM(array_interventions[i], x, array_net_x)]))
    else:
      # Plot the empirical distribution of DCM and true target
      array_DCM_samples = np.vstack((array_DCM_samples, np.array([sample_outcome_do_cause_DCM(array_interventions[i], x, array_net_x)])))
      # Plot the empirical distribution of BDCM and true target
      array_BDCM_samples = np.vstack((array_BDCM_samples, np.array([sample_outcome_do_cause_BDCM(array_interventions[i], x, array_net_x)])))
    array_array_DCM_BDCM_samples = np.array([array_DCM_samples, array_BDCM_samples])

  return array_array_DCM_BDCM_samples