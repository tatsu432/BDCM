import numpy as np
import torch
import random
import matplotlib.pyplot as plt
import conf

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