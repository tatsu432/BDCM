# Import libraries
import numpy as np
import matplotlib.pyplot as plt
import random
import torch
import torch.nn.functional
import torch.utils.data

# Warnings
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

plt.style.use('ggplot')


import conf
from func import set_seed, normalize, DEC, create_array_array_MMD, train_and_plot_neural_net, create_alpha_t_train_for_x, save_array, create_input_for_NN, calculate_overall_MMD
# from sample import sample_array_u
from sample import sample_array_u_and_x, true_sample, sample_t_and_eps_for_x

from tqdm import tqdm

plt.ioff()  # 対話モードを無効にする



def SCM4(structural_eq, simple_or_complex):

    name_of_folder = "SCM4"

    # Define the number of endogenous or exogenous variables in SCM
    d = 10

    # index for cause variable X_i = 9
    ind_cause = 8

    # index for result variable X_i = 10
    ind_result = 9

    # the array of the list of the indexes of the parent node in DAG
    array_list_parent_ind = [[], [], [1], [2], [3], [4], [3], [4], [5, 8], [6, 7, 9]]

    # Define the function to sample X_4 n times when we intervene to X_1 in DCM
    # Input: the number of samples we want to obtain and the value of the intervention
    # Output: the vector of the samples from the target distribution
    def sample_outcome_do_cause_DCM(intervened_value, x, array_net_x):
        # Initialize the sample list by just zero vector
        x_outcome_DDIM_list = np.zeros(conf.n_sample_DCM)

        # Iteratively sample from the target distribution
        for i in range(conf.n_sample_DCM):
            # Sample by the empirical distribution
            x3_sampled = random.choice(x[2])
            # Sample by the empirical distribution
            x4_sampled = random.choice(x[3])
            # Concatenate the parents and nodes which satisfy the backdoor criterion
            x7_parents = np.array([x3_sampled])
            # Sample X_7 by using the decoder function
            x7_sampled = DEC(x7_parents, 0, array_net_x)
            # Concatenate the parents and nodes which satisfy the backdoor criterion
            x8_parents = np.array([x4_sampled])
            # Sample X_8 by using the decoder function
            x8_sampled = DEC(x8_parents, 1, array_net_x)
            # Set X_9 to the intervened value
            x9_sampled = intervened_value
            # Concatenate the parents and nodes which satisfy the backdoor criterion
            x10_parents = np.array([x7_sampled, x9_sampled])
            # Sample X_2 by using the decoder function
            x10_sampled = DEC(x10_parents, 2, array_net_x)
            # Add the sampled value to the list
            x_outcome_DDIM_list[i] = x10_sampled

        return x_outcome_DDIM_list


    # Define the function to sample X_4 n times when we intervene to X_1 in BDCM
    # Input: the number of samples we want to obtain and the value of the intervention
    # Output: the vector of the samples from the target distribution
    def sample_outcome_do_cause_BDCM(intervened_value, x, array_net_x):
        # Initialize the sample list by just zero vector
        x_outcome_DDIM_list = np.zeros(conf.n_sample_DCM)

        # Iteratively sample from the target distribution
        for i in range(conf.n_sample_DCM):
            # Sample by the empirical distribution
            x3_sampled = random.choice(x[2])
            # Sample by the empirical distribution
            x4_sampled = random.choice(x[3])
            # Concatenate the parents and nodes which satisfy the backdoor criterion
            x7_parents = np.array([x3_sampled])
            # Sample X_7 by using the decoder function
            x7_sampled = DEC(x7_parents, 0, array_net_x)
            # Concatenate the parents and nodes which satisfy the backdoor criterion
            x8_parents = np.array([x4_sampled])
            # Sample X_8 by using the decoder function
            x8_sampled = DEC(x8_parents, 1, array_net_x)
            # Set X_9 to the intervened value
            x9_sampled = intervened_value
            # Concatenate the parents and nodes which satisfy the backdoor criterion
            x10_parents = np.array([x3_sampled, x4_sampled, x9_sampled])
            # Sample X_2 by using the decoder function
            x10_sampled = DEC(x10_parents, 3, array_net_x)
            # Add the sampled value to the list
            x_outcome_DDIM_list[i] = x10_sampled

        return x_outcome_DDIM_list

    # Initialized the array to save the output of each iterations over seeds
    array_array_MMD = []

    for s in tqdm(range(conf.num_seeds)):
        # Set the seed
        set_seed(s)

        # Define the array of interneved values
        # 10 interventinos with the intervened value ranging from -3 to 3 linearly
        array_interventions = np.random.uniform(conf.lowest_intervention, conf.highest_intervention, conf.num_interventions)

        # # Sample exogenous nodes U_i ~ N(mu, sigma)
        # # Sample endogenous nodes X_i by the structural equations
        array_u, x = sample_array_u_and_x(d, structural_eq)

        # # Generate t ~ Unif([T])
        # # Generate \epsilon ~ N(mu, sigma)
        t_for_x, epsilon_for_x = sample_t_and_eps_for_x(d)

        # # Get the alpha_t for training
        alpha_t_train_for_x = create_alpha_t_train_for_x(d, t_for_x)


        # the nodes for which we use DEC
        array_titles = np.array(["X_7", "X_8", "X_10 (DCM)", "X_10 (BDCM)"])

        # Define the array of the index for epsilon for the neural networks (index - 1)
        array_index_for_epsilon = np.array([6, 7, 9, 9])

        # Define the array of the numbers of the inputs for the neural networks (2 + number of parents or adjustment set)
        array_num_input_for_nn = np.array([3, 3, 4, 5])

        # Define the array of the parents or the adjustment set for each DEC (index - 1)
        parent = [[2], [3], [6, 8], [2, 3, 8]]

        # Create the input for neural network
        array_input_x = create_input_for_NN(array_num_input_for_nn, array_index_for_epsilon, alpha_t_train_for_x, x, epsilon_for_x, parent, t_for_x)


        # """Train the Neural Network"""
        array_net_x = train_and_plot_neural_net(array_input_x, epsilon_for_x, array_index_for_epsilon, array_num_input_for_nn, array_titles, conf.flag_plot_nn_train)

        # # Create the array that save the intervened value, samples from DCM, samples from BDCM
        # Get the samples from DCM and BDCM
        array_array_DCM_BDCM_samples = save_array(array_interventions, sample_outcome_do_cause_DCM, sample_outcome_do_cause_BDCM, x, array_net_x)

        create_array_array_MMD(array_interventions, array_array_DCM_BDCM_samples, true_sample, d, structural_eq, ind_cause, ind_result, array_u, array_array_MMD, name_of_folder, s, simple_or_complex, conf.flag_print_each_MMD, conf.flag_show_plot)

    calculate_overall_MMD(array_array_MMD)