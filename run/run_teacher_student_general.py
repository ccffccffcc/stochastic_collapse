import numpy as np
from tqdm import tqdm
import torch
import pickle
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int)
args = parser.parse_args()


def generate(n, m, sigma):
    Ux, _ = np.linalg.qr(np.random.randn(n, n))
    Vx, _ = np.linalg.qr(np.random.randn(m, m))
    Sx = np.zeros((n, m))
    for i, val in enumerate(sigma):
        Sx[i, i] = val
    return Ux.dot(Sx).dot(Vx.T)


# V: n x n
# U: n x m


def forward(U, V, x):
    return x @ V @ U


def loss(U, V, x, y):
    yhat = forward(U, V, x)
    return 0.5 * torch.sum(torch.square(yhat - y))


def loss_test(U, V, W):
    return torch.mean(torch.square(V @ U - W))


def gradient(U, V, x, y, sigma=0, lamb=0):
    yhat = forward(U, V, x)
    residual = yhat - y + sigma * torch.randn(y.size()).cuda()
    gV = (U @ residual.T @ x).T + lamb * V
    gU = (residual.T @ x @ V).T + lamb * U
    return gU, gV


def gradient_inputnoise(U, V, x, y, sigma=0, lamb=0):
    x = x + sigma * torch.randn(x.size()).cuda()
    yhat = forward(U, V, x)
    residual = yhat - y
    gV = (U @ residual.T @ x).T + lamb * V
    gU = (residual.T @ x @ V).T + lamb * U
    return gU, gV


# Data generation
seed = 0
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
np.random.seed(0)
# reproducibility
N, n, m, d = 1024, 64, 64, 8
W = torch.from_numpy(generate(n, m, np.linspace(0.5, 1.0, d))).float().cuda()
X = torch.rand(N, n).cuda() / np.sqrt(n)
X_test = torch.rand(N, n).cuda() / np.sqrt(n)

sigma = 0.5
y = X @ W + sigma * torch.randn(N, m).cuda() / np.sqrt(m)
y_true = X @ W
y_test = X_test @ W + sigma * torch.randn(N, m).cuda() / np.sqrt(m)

V0 = torch.rand(n, n).cuda() / 2
U0 = torch.rand(n, m).cuda() / 2

tt = args.seed
np.random.seed(tt)

eta, steps = 0.0, 100001
Ltrain = []
Us_list = []
Vs_list = []
Ws_list = []
Ltest = []

n_batch_list = [1, 2, 4, 8, 16, 1024]
sigma = 0
for n_batch in n_batch_list:
    print(n_batch)
    lr = eta
    ltrain = []
    ltest = []
    Us = []
    Vs = []
    Ws = []
    V = V0.clone()
    U = U0.clone()

    for i in tqdm(range(steps)):
        if i < 10000:
            lr += 3e-4

        if i in [50000, 53000]:
            lr *= 0.1

        loss_train = loss(U, V, X, y)
        if loss_train.item() > 10**10:
            print("Nan")
        ltrain.append(loss_train.item())
        ltest.append(loss_test(U, V, W).item())
        if i % 100 == 0:
            Us.append(U.cpu().numpy())
            Vs.append(V.cpu().numpy())
            Ws.append((V @ U).cpu().numpy())
        if n_batch == 1:
            ind = np.random.randint(0, N, 1)
        else:
            ind = np.random.choice(np.arange(N), n_batch, replace=False)
        vU, vV = gradient(U, V, X[ind], y[ind], sigma=sigma)  # Effect of Label Noise
        #         vU, vV = gradient(U, V, XX, lamb=sigma, sigma=0) # Effect of L2 Regularization
        Uvel = vU / n_batch
        Vvel = vV / n_batch

        U -= lr * Uvel
        V -= lr * Vvel
    Us_list.append(Us)
    Vs_list.append(Vs)
    Ws_list.append(Ws)

    Ltrain.append(ltrain)
    Ltest.append(ltest)

a = {"Ltrain": Ltrain, "Ltest": Ltest, "Us_list_list": Us_list, "Vs_list_list": Vs_list}

with open("exps/Teacher_student_setup_SGD_warmup_%d.pickle" % tt, "wb") as handle:
    pickle.dump(a, handle, protocol=pickle.HIGHEST_PROTOCOL)
