import torch
import numpy as np
sigma_0 = 1
d = 3  # dimension of each sample
s = 100  # number of samples
k = 2  # dimension of the task subspaces

N_L = 2 # tasks per cluster
N_L = min(N_L, int(np.floor(d/k))) # tasks per cluster


M = 2 # number of experts
L = 2 # number of task clusters
N = N_L*L # total number of unique tasks


alpha=1e0 # scalar of the auxilary loss function
beta=1e0 # scalar of the align loss function (load balancing)
gamma=0e2 # contrastive loss scale
eta=1e-2 # gate learning rate
w_eta = 1e-4 # expert learning rate

# first_task_arrival = 100
# time_to_new = 300
# task_split = 4
# T= first_task_arrival + (task_split - 1)*time_to_new# 2000 # number of task arrivals

T = 300

epochs = 20

#device = torch.device("cuda" if torch.cuda.is_available() else 
#                      "mps" if torch.backends.mps.is_available() else "cpu")

np.random.seed(42)