import numpy as np


# Define the function to normalize
def normalize(x):
    return (x - np.mean(x)) / np.std(x)






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





# Create the array that save the intervened value, samples from DCM, samples from BDCM
# Input: array of intervened values
# Output: DCM and BDCM samples
def save_array(array_interventions):
    # Get the number of interventions
    num_intervention = np.size(array_interventions)

    # Initialize the array of samples for DCM and BDCM
    array_DCM_samples = np.array([])
    array_BDCM_samples = np.array([])

    # for each intervention
    for i in range(num_intervention):
      if np.any(array_DCM_samples) == False:
        # Plot the empirical distribution of DCM and true target
        array_DCM_samples = np.append(array_DCM_samples, np.array([sample_outcome_do_cause_DCM(array_interventions[i])]))
        # Plot the empirical distribution of BDCM and true target
        array_BDCM_samples = np.append(array_BDCM_samples, np.array([sample_outcome_do_cause_BDCM(array_interventions[i])]))
      else:
        # Plot the empirical distribution of DCM and true target
        array_DCM_samples = np.vstack((array_DCM_samples, np.array([sample_outcome_do_cause_DCM(array_interventions[i])])))
        # Plot the empirical distribution of BDCM and true target
        array_BDCM_samples = np.vstack((array_BDCM_samples, np.array([sample_outcome_do_cause_BDCM(array_interventions[i])])))
      array_array_DCM_BDCM_samples = np.array([array_DCM_samples, array_BDCM_samples])

    return array_array_DCM_BDCM_samples