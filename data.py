import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from experiment_settings import *


class RegressionDataset(Dataset):
    def __init__(self, X, y):
        # Convert data to torch tensors
        self.X = torch.tensor(np.ascontiguousarray(X), dtype=torch.float32)
        self.y = torch.tensor(np.ascontiguousarray(y), dtype=torch.float32)

    def __len__(self):
        # Return the size of the dataset
        return len(self.X)

    def __getitem__(self, idx):
        # Return a single sample and label pair
        return self.X[idx], self.y[idx]


# generate (X,y) pair for one subspace U_t and w_t
def generate_samples(U, w, n, sigma=0.05):
    alpha = np.random.standard_normal((n, np.shape(U)[1]))  # random coefficients to generate x_i
    X = alpha @ U.T
    # noise = sigma * randGen.standard_normal(n)
    y = X @ w.T  #+ noise
    return X, y

def create_datasets(totalTasks, numTasks, numSamples, baseDim, subDim, sigma=10, batchSize=32, randSeed=73827646):
    """ Create datasets for orthogonal tasks. Subspaces are of fixed dimension k but they must be orthgonal"""
    # Instantiate random number generator
    # rng = np.random.default_rng(randSeed)

    d = baseDim  # base dimension
    k = subDim  # U dimension

    n = numSamples  # total samples per task

    # Generate a random orthogonal basis for R^d
    U = np.random.standard_normal(size=(d, d))
    U, _ = np.linalg.qr(U)  # U has orthogonal basis now

    # Create orthogonal subspaces for each task
    Us = [U[:, i * k:(i + 1) * k] for i in range(numTasks)]
    # Us = np.array(Us)

    # Generate random parameters for each task
    ws = [sigma*np.random.standard_normal(size=(1,d)) for _ in range(numTasks)]
    # ws = np.array(ws)

    # Create datasets for each task using different permutations
    # torch.manual_seed(randSeed)

    task_record=[]
    if task_split <= numTasks:
        for i in range(totalTasks):
            if i > first_task_arrival:
                for j in range(0, int((totalTasks - first_task_arrival)/time_to_new)):
                    if np.floor((i - first_task_arrival)/time_to_new) == j:
                        task_record.append(np.random.randint(0,np.floor((j+1+1)*numTasks/task_split)))
                    
            else:
                task_record.append(np.random.randint(0,np.floor(numTasks/task_split)))
            # for j in range(task_split):
            #     if np.floor(i/(totalTasks/task_split)) == j:
            #         task_record.append(np.random.randint(0,int(min(np.floor((j+1)*numTasks/task_split), np.floor(2*numTasks/task_split)))))
    else:
        task_record = np.random.randint(0,numTasks, totalTasks)
    task_record = np.array(task_record)

    train_datasets = [
        RegressionDataset(
            *generate_samples(Us[task_record[idx]], ws[task_record[idx]], n, sigma=sigma)
        )
        for idx in range(totalTasks)
    ]
    test_datasets = [
        RegressionDataset(
            *generate_samples(Us[task_record[idx]], ws[task_record[idx]], n, sigma=sigma)
        )
        for idx in range(totalTasks)
    ]

    train_loaders = [
        DataLoader(dataset, batch_size=batchSize, shuffle=False)
        for dataset in train_datasets
    ]
    test_loaders = [
        DataLoader(dataset, batch_size=batchSize, shuffle=False)
        for dataset in test_datasets
    ]

    return train_loaders, test_loaders, Us, ws, task_record


def create_datasets_2(totalTasks, numClusters, perCluster, numSamples, baseDim, subDim, sigma=10, batchSize=32):
    """ Create datasets for orthogonal task clusters. Subspaces are of fixed dimension k but they must be orthgonal"""
    # Instantiate random number generator
    # rng = np.random.default_rng(randSeed)

    d = baseDim  # base dimension
    k = subDim  # U dimension

    n = numSamples  # total samples per task

    L = numClusters
    T = totalTasks

    N_L = perCluster # tasks per cluster
    N = N_L*L # total number of unique tasks

    Us = []
    ws = []
    for i in range(L):
        # Generate a random orthogonal basis for R^d
        U = np.random.standard_normal(size=(d, d))
        U, _ = np.linalg.qr(U)
        # Create orthogonal subspaces for each task
        for j in range(N_L):
            Us.append(U[:, j * k:(j + 1) * k])
        # Generate random parameters for each cluster
        ws.append(sigma*np.random.standard_normal(size=(1,d)))

    # create task record
    task_record = np.random.randint(0,N, T)


    train_datasets = [
        RegressionDataset(
            *generate_samples(Us[task_record[idx]], ws[int(task_record[idx]/N_L)], n, sigma=sigma)
        )
        for idx in range(T)
    ]
    test_datasets = [
        RegressionDataset(
            *generate_samples(Us[task_record[idx]], ws[int(task_record[idx]/N_L)], n, sigma=sigma)
        )
        for idx in range(T)
    ]

    train_loaders = [
        DataLoader(dataset, batch_size=batchSize, shuffle=False)
        for dataset in train_datasets
    ]
    test_loaders = [
        DataLoader(dataset, batch_size=batchSize, shuffle=False)
        for dataset in test_datasets
    ]

    return train_loaders, test_loaders, Us, ws, task_record

