import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from data import *

from experiment_settings import *
    
def train_loss(y_hat, y_t):
        # s = X_t.shape[0]  # the number of samples
        # print(s)
        loss = 1/len(y_t)*torch.norm(y_hat - y_t, p=2) ** 2
        return loss
    
# modified version of the load loss function (only considering last gate output selection)
def load_loss_mod(expert_usage_history_t, expert_selected_t, gate_output):
    expert_usage_t = torch.sum(expert_usage_history_t, dim=1)
    aux_loss = expert_usage_t[expert_selected_t]*gate_output[expert_selected_t]
    return M*aux_loss


def align_loss(experts, gate_output, X_t, y_t):
    al_loss = 0
    for idx, expert in enumerate(experts):
        al_loss += gate_output[idx]*train_loss(expert(X_t), y_t)
        # print(f"{idx}: {train_loss(expert(X_t), y_t)}")
    return al_loss/len(experts)

def contra_loss(experts, gating_network, topk_index, X_t, y_t, X_t_1, y_t_1, alpha=1, beta=150, eps=1e-3):
    contra_loss = 0
    gate_output_t = gating_network(X_t).mean(dim=0)
    gate_output_t_1 = gating_network(X_t_1).mean(dim=0)
    y = torch.dot(gate_output_t,gate_output_t_1)
    expert_t_loss = train_loss(y_t, experts[topk_index](X_t))
    expert_t_1_loss = train_loss(y_t_1, experts[topk_index](X_t_1))
    x = torch.abs(expert_t_loss-expert_t_1_loss)

    log_diff = torch.square(torch.log(alpha*x + eps) - torch.log(beta*y + eps))
    contra_loss = torch.exp(-log_diff)
    return contra_loss


# for now don't worry about growing (thus remove distribution_shifts_t)
def gen_error(moe, Us, ws, distribution_shifts_t=None, t=0):
    if t == 0:
        return 0,0,0,0
    
    if distribution_shifts_t is None:
        N_t = N
    else:
        task_split = len(distribution_shifts_t)
        
        if task_split <= N:
            for idx in range(len(distribution_shifts_t)):
                if t - distribution_shifts_t[idx] < 0:
                    N_t = int(np.floor((idx+1)*N/task_split))
                    break
                else:
                    N_t = N
        else:
            N_t = N

    M = len(moe.experts)
    gen_loss = torch.tensor(0, dtype=torch.float32).to(device)
    align_loss = torch.tensor(0, dtype=torch.float32).to(device)
    model_error= torch.tensor(0, dtype=torch.float32).to(device)
    proj_model_error = torch.tensor(0, dtype=torch.float32).to(device)
    num_trials = 2
    for _ in range(num_trials):
        for idx in range(N_t):


            # Generate Data
            w_index=idx
            wn = ws[int(w_index/N_L)]
            vn = Us[w_index]
            Xt, yt = generate_samples(vn, wn, s)
            dataset = RegressionDataset(Xt,yt)
            dataloader = DataLoader(dataset, batch_size=s, shuffle=False)

            with torch.no_grad():
                for Xt, yt in dataloader:
                    Xt, yt = Xt.to(device), yt.to(device)

                    wn = torch.tensor(wn, dtype=torch.float32).to(device)
                    vn = torch.tensor(vn, dtype=torch.float32).to(device)

                    moe_output, softmax_value, expert_seleted= moe(Xt)

                    # take mean of softmax value across batch
                    softmax_value = softmax_value
                    
                    gate_uncertain = 0
                    if torch.sum(softmax_value == torch.max(softmax_value)) > 1:
                        gate_uncertain = 1
                        rand_idx = np.random.randint(0, M)

                    for i in range(0,M):
                        tr_loss = 1/s*torch.norm(yt - moe.experts[i](Xt), p=2)**2
                        align_loss += softmax_value[i]*tr_loss
                        if i == expert_seleted and  not gate_uncertain:
                            # print(f"{idx}: {tr_loss}")
                            gen_loss += tr_loss
                            # model_error += torch.norm(wn - moe.experts[i].fc.weight)**2
                            # proj_model_error += torch.norm(torch.dot(wn[0] - moe.experts[i].fc.weight[0], vn[:,0])*vn[:,0])**2
                        elif gate_uncertain and i == rand_idx:
                            # print(f"{idx}: {tr_loss}")
                            gen_loss += tr_loss
                            # model_error += torch.norm(wn - moe.experts[i].fc.weight)**2
                            # proj_model_error += torch.norm(torch.dot(wn[0] - moe.experts[i].fc.weight[0], vn[:,0])*vn[:,0])**2
                        
                    
                
                
                
    return gen_loss.item()/N_t/num_trials, align_loss.item()/N_t/num_trials, model_error.item()/N_t/num_trials, proj_model_error.item()/N_t/num_trials
     
