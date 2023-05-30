import numpy as np
from tqdm import tqdm
import scipy.stats
import torch
import pickle
import argparse
import wandb

device = "cuda"

wandb.init()

parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int)
args = parser.parse_args()


def generate(n, m, sigma):
    Ux = scipy.stats.ortho_group.rvs(n)
    Vx = scipy.stats.ortho_group.rvs(m)
    Sx = np.zeros((n, m))
    for i, val in enumerate(sigma):
        Sx[i, i] = val
    return Ux, Vx, Ux @ Sx @ Vx


def forward(U, V, x):
    return x @ V[0] @ torch.diag(V[1]) @ torch.diag(U[1]) @ U[0]


def loss(U, V, x, y):
    yhat = forward(U, V, x)
    return 0.5 * torch.sum(torch.square(yhat - y))


def loss_test(U, V, W):
    return torch.mean(torch.square(V[0] @ torch.diag(V[1]) @ torch.diag(U[1]) @ U[0] - W))


def gradient(U, V, sx, sigma=0):
    su, sv = U[1], V[1]
    noise = sigma * torch.randn(size=(len(su),)).to(device)
    residual = sv * su - sx + noise
    gV = residual * su
    gU = residual * sv
    return gV, gU


##Data generation
np.random.seed(0)  # reproducibility
N, n, m, d = 1024, 64, 64, 8
_, _, W = generate(n, m, np.linspace(0.5, 1, d))
_, _, X = generate(N, n, np.ones(n))

sigma = 0.5
y = X @ W + sigma * np.random.randn(N, m) / np.sqrt(m)
y_true = X @ W

Vx, Sx, Ux = np.linalg.svd(X.T @ y)

W = torch.from_numpy(W).float().to(device)
X = torch.from_numpy(X).float().to(device)
y = torch.from_numpy(y).float().to(device)
y_true = torch.from_numpy(y_true).float().to(device)
Vx = torch.from_numpy(Vx).float().to(device)
Sx = torch.from_numpy(Sx).float().to(device)
Ux = torch.from_numpy(Ux).float().to(device)

from multiprocessing import Pool as p

eta, steps = 3.0, 60001

Ltrain_list = []
Us_list_list = []
Vs_list_list = []
Ltest_list = []

torch.manual_seed(args.seed)
np.random.seed(args.seed)

with torch.no_grad():
    Ltrain = []
    Us_list = []
    Vs_list = []
    Ltest = []

    n_batch = len(X)

    sigmas = [0, 5, 10, 15, 20]
    for sigma in sigmas:
        lr = eta
        ltrain = []
        ltest = []
        Us = []
        Vs = []
        V = [Vx.clone(), torch.ones(n).to(device)]
        U = [Ux.clone(), torch.ones(m).to(device)]

        for i in tqdm(range(steps)):
            if i in [50000]:
                lr *= 0.1
            if i % 10 == 0:
                ltrain.append(loss(U, V, X, y).item())
                ltest.append(loss_test(U, V, W).item())

                #
                Us.append(U[1].detach().cpu().numpy())
                Vs.append(V[1].detach().cpu().numpy())

            vV, vU = gradient(U, V, Sx, sigma=sigma)  # Effect of Label Noise
            Uvel = vU / n_batch
            Vvel = vV / n_batch

            U[1] -= lr * Uvel
            V[1] -= lr * Vvel
        Us_list.append(Us)
        Vs_list.append(Vs)

        Ltrain.append(ltrain)
        Ltest.append(ltest)

a = {"Ltrain": Ltrain, "Ltest": Ltest, "Us_list_list": Us_list, "Vs_list_list": Vs_list}

with open("exps/Teacher_student_setup_SGD_diagonal_f_%d.pickle" % args.seed, "wb") as handle:
    pickle.dump(a, handle, protocol=pickle.HIGHEST_PROTOCOL)

wandb.finish()