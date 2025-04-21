import torch
import numpy as np

from data import *
from experiment_settings import *
from loss import *

training_loader, testing_loader, Us, ws, task_record = create_datasets_2(totalTasks=T, numClusters=L, perCluster=N_L, numSamples=s, baseDim=d, subDim=k, batchSize=int(s/1), sigma=20)

class SubspaceClassifierTorch:
    def __init__(self, N, k, d, experts):
        self.N = N
        self.k = k
        self.d = d
        self.experts = experts
        self.known_subspaces = []
        self.expert_guess_matrix = np.tile(np.arange(0, N), (N, 1)).tolist()
        self.true_expert = np.zeros(N) - 1
        self.guess = []
        self.rewards = []

    def _extract_basis(self, X):
        X_centered = X - X.mean(dim=0, keepdim=True)
        cov = X_centered.T @ X_centered / (X_centered.shape[0] - 1)
        eigvals, eigvecs = torch.linalg.eigh(cov)
        return eigvecs[:, -self.k:].T

    def _match_existing_subspace(self, new_basis):
        for i, basis in enumerate(self.known_subspaces):
            combined = torch.cat([new_basis, basis], dim=0)
            if torch.linalg.matrix_rank(combined) <= self.k:
                return i
        return None

    def _eliminate_class(self, subspace_idx, wrong_class):
        if wrong_class in self.expert_guess_matrix[subspace_idx]:
            self.expert_guess_matrix[subspace_idx].remove(wrong_class)

    def _reward(self, X, y, expert_selected):
        expert = torch.from_numpy(self.experts[expert_selected]).to(torch.float32)
        expert_output = X @ expert.T
        loss = train_loss(expert_output, y)
        return 1 if loss < 1e-9 else 0

    def process(self, X_t, y_t):
        basis = self._extract_basis(X_t)
        subspace_idx = self._match_existing_subspace(basis)

        if subspace_idx is not None and self.true_expert[subspace_idx] >= 0:
            guess = int(self.true_expert[subspace_idx])
        else:
            if subspace_idx is not None:
                guess = int(np.random.choice(self.expert_guess_matrix[subspace_idx]))
            else:
                self.known_subspaces.append(basis)
                subspace_idx = len(self.known_subspaces) - 1
                guess = int(np.random.choice(self.expert_guess_matrix[subspace_idx]))

        reward = self._reward(X_t, y_t, guess)

        self.rewards.append(reward)
        self.guess.append(guess)

        if not reward:
            self._eliminate_class(subspace_idx, guess)
        else:
            self.true_expert[subspace_idx] = guess
            for idx in range(self.N):
                if guess in self.expert_guess_matrix[idx]:
                    self._eliminate_class(idx, guess)

        return subspace_idx, guess, reward

# Example run
clf = SubspaceClassifierTorch(N=N, k=k, d=d, experts=ws)
print(f"N = {N}")
for t in range(T):
    task_training_loader = training_loader[t]
    for _, data in enumerate(task_training_loader):
        X_t, y_t = data
        idx, guess, reward = clf.process(X_t, y_t)

print(f"Number of mistakes: {T - np.sum(clf.rewards)}")

import matplotlib.pyplot as plt

y_lim=None

plt.figure(9)
plt.plot(np.cumsum(clf.rewards))
plt.xlabel("Rounds")
plt.ylabel("Reward (Inverse Hamming Loss)")
plt.ylim(0, y_lim)
plt.show()

plt.figure(9)
plt.plot(np.arange(T) - np.cumsum(clf.rewards))
plt.xlabel("Rounds")
plt.ylabel("Regret (Inverse Hamming Loss)")
plt.ylim(0, y_lim)
plt.show()