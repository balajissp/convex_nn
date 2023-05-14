from datetime import datetime as dt
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from pathlib import Path

import time
import torch
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable
import torch.nn as nn
import argparse
import random
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-skipGD', dest='GD', action='store_false')
    parser.add_argument('-skipCVX', dest='CVX', action='store_false')
    parser.add_argument('--n_epochs', nargs=2, type=int, default=[100,100])
    parser.add_argument('--batch_size', type=int, default=5000)
    parser.add_argument('--solver_cvx', default="sgd", choices=["sgd", "adam", "adagrad", "adadelta", "LBFGS"])
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', type=str, default='cuda', choices=['cuda', 'cpu'])
    return parser.parse_args()


class FCNetwork(nn.Module):
    def __init__(self, h, num_classes=10, input_dim=3072):
        self.num_classes = num_classes
        super(FCNetwork, self).__init__()
        self.layer1 = nn.Sequential(nn.Linear(input_dim, h, bias=False), nn.ReLU())
        self.layer2 = nn.Linear(h, num_classes, bias=False)

    def forward(self, x):
        x = x.reshape(x.size(0), -1)
        out = self.layer2(self.layer1(x))
        return out


class PrepareData3D(Dataset):
    def __init__(self, x, y, z):
        if not torch.is_tensor(x):
            self.X = torch.from_numpy(x)
        else:
            self.X = x

        if not torch.is_tensor(y):
            self.y = torch.from_numpy(y)
        else:
            self.y = y

        if not torch.is_tensor(z):
            self.z = torch.from_numpy(z)
        else:
            self.z = z

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx], self.z[idx]


def generate_conv_sign_patterns(data, P, c, p1, p2):
    fs = 3
    d = c * p1 * p2
    ind1 = np.random.randint(0, p1 - fs + 1, size=P)
    ind2 = np.random.randint(0, p2 - fs + 1, size=P)
    u1p = np.zeros((c, p1, p2, P))
    u2 = np.random.normal(0, 1, size=(c, fs, fs, P))
    for i, (i1, i2) in enumerate(zip(ind1, ind2)):
        u1p[:, i1:i1 + fs, i2:i2 + fs, i] = u2[:, :, :, i]
    u1 = u1p.reshape(d, P)
    sampled_sign_patterns = (np.matmul(data, u1) >= 0).T

    print(f"{P} unique sign patterns generated.")
    return sampled_sign_patterns, u1.T


def generate_sign_patterns(data, P):
    # generate sign patterns
    n, d = data.shape
    umat = np.random.normal(0, 1, (d, P))
    sampled_sign_pattern_mat = (np.matmul(data, umat) >= 0)
    sign_pattern_list = sampled_sign_pattern_mat[:, :P].T
    u_vector_list = umat[:, :P].T
    print(f"{P} sign patterns generated")
    return sign_pattern_list, u_vector_list


def one_hot(labels, num_classes=10):
    y = torch.eye(num_classes).to(device)
    return y[labels.long()]


# =====================================STANDARD NON-CONVEX NETWORK=====================================
def loss_func_primal(yhat, y, model, beta):
    loss = 0.5 * torch.norm(yhat - y).square()

    # l2 norm on first layer weights, l1 squared norm on second layer
    for layer, p in enumerate(model.parameters()):
        if layer == 0:
            loss += beta / 2 * torch.norm(p).square()
        else:
            loss += beta / 2 * torch.norm(p, 1, dim=0).square().sum()

    return loss


def validation_primal(model, testloader, beta, test_len):
    test_loss = 0
    test_correct = 0
    for _x, _y in testloader:
        _x = _x.float().to(device)
        _y = _y.float().to(device)
        yhat = model(_x).float()
        loss = loss_func_primal(yhat, one_hot(_y), model, beta)
        test_loss += loss.cpu().detach().numpy()
        test_correct += torch.eq(torch.argmax(yhat, dim=1), torch.squeeze(_y)).float().sum()

    return test_loss / test_len, test_correct / test_len


# solves nonconvex problem
def sgd_solver_pytorch_v2(ds_train, ds_test, num_epochs, num_neurons, beta, learning_rate, batch_size, solver_type,
                          schedule, LBFGS_param, num_classes=10, d_in=3 * 1024, test_len=10000,
                          train_len=50000):
    # D_in is input dimension, H is hidden dimension, D_out is output dimension.
    # create the model
    model = FCNetwork(h=num_neurons, num_classes=num_classes, input_dim=d_in).to(device)

    if solver_type == "sgd":
        optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
    elif solver_type == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    elif solver_type == "adagrad":
        optimizer = torch.optim.Adagrad(model.parameters(), lr=learning_rate)
    elif solver_type == "adadelta":
        optimizer = torch.optim.Adadelta(model.parameters(), lr=learning_rate)
    elif solver_type == "LBFGS":
        optimizer = torch.optim.LBFGS(model.parameters(), history_size=LBFGS_param[0], max_iter=LBFGS_param[1])
    else:
        raise ValueError('solver_type should be one of sgd/adam/adagrad/adadelta/LBFGS.\nGot ' + solver_type)

    # arrays for saving the loss and accuracy
    losses = np.zeros(num_epochs)
    accs = np.zeros(num_epochs)
    times = np.zeros(num_epochs)
    losses_test = np.zeros((num_epochs + 1))
    accs_test = np.zeros((num_epochs + 1))
    start_time = time.time()

    losses_test[0], accs_test[0] = validation_primal(model, ds_test, beta, test_len)  # loss on the entire test set
    if schedule == 1:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, verbose=True, factor=0.5, eps=1e-12)
    elif schedule == 2:
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, 0.99)
    else:
        scheduler = None

    for i in tqdm(range(num_epochs)):
        for _x, _y in ds_train:
            # =========make input differentiable=======================
            _x = Variable(_x).to(device)
            _y = Variable(_y).to(device)

            # ========forward pass=====================================
            yhat = model(_x).float()

            loss = loss_func_primal(yhat, one_hot(_y), model, beta) / len(_y)
            correct = torch.eq(torch.argmax(yhat, dim=1), torch.squeeze(_y)).float().sum() / len(_y)

            optimizer.zero_grad()  # zero the gradients on each pass before the update
            loss.backward()  # backpropagate the loss through the model
            optimizer.step()  # update the gradients w.r.t the loss

            losses[i] += loss.cpu().detach().numpy() * len(_y)  # loss on the minibatch
            accs[i] += correct * len(_y)

        times[i] = time.time() - start_time
        losses[i] /= train_len
        accs[i] /= train_len
        # get test loss and accuracy
        losses_test[i + 1], accs_test[i + 1] = validation_primal(model, ds_test, beta, test_len)
        if scheduler is not None:
            scheduler.step(losses[i])

        print(f"Epoch [{i + 1}/{num_epochs}], loss: {losses[i]:.3f} acc: {accs[i]:.3f}, "
              f"test loss: {losses_test[i + 1]:.3f} test acc: {accs_test[i + 1]:.3f}")

    return losses, accs, losses_test, accs_test, times, model


# =====================================CONVEX NETWORK=====================================
class CustomCVXLayer(torch.nn.Module):
    def __init__(self, d, num_neurons, num_classes=10):
        """
        In the constructor we instantiate two nn.Linear modules and assign them as
        member variables.
        """
        super(CustomCVXLayer, self).__init__()

        # P x d x C
        self.v = torch.nn.Parameter(data=torch.zeros(num_neurons, d, num_classes), requires_grad=True)
        self.w = torch.nn.Parameter(data=torch.zeros(num_neurons, d, num_classes), requires_grad=True)

    def forward(self, x, sign_patterns):
        sign_patterns = sign_patterns.unsqueeze(2)
        x = x.view(x.shape[0], -1)  # n x d
        Xv_w = torch.matmul(x, self.v - self.w)  # P x N x C

        # for some reason, the permutation is necessary. not sure why
        DXv_w = torch.mul(sign_patterns, Xv_w.permute(1, 0, 2))  # N x P x C
        y_pred = torch.sum(DXv_w, dim=1, keepdim=False)  # N x C

        return y_pred


def get_nonconvex_cost(y, model, _x, beta):
    _x = _x.view(_x.shape[0], -1)
    Xv_relu = torch.nn.functional.relu(torch.matmul(_x, model.v))
    Xw_relu = torch.nn.functional.relu(torch.matmul(_x, model.w))
    prediction_w_relu = torch.sum(Xv_relu - Xw_relu, dim=0, keepdim=False)

    # prediction cost + regularization cost
    return 0.5 * torch.norm(prediction_w_relu - y).square() + \
           beta * torch.sum(torch.norm(model.v, dim=1).square()) + \
           beta * torch.sum(torch.norm(model.w, p=1, dim=1).square())


def loss_func_cvxproblem(yhat, y, model, _x, sign_patterns, beta, rho):
    _x = _x.view(_x.shape[0], -1)
    # term 1
    loss = 0.5 * torch.norm(yhat - y) ** 2
    # term 2
    loss = loss + beta * torch.sum(torch.norm(model.v, dim=1)) + beta * torch.sum(torch.norm(model.w, dim=1))
    # term 3
    sign_patterns = sign_patterns.unsqueeze(2)  # N x P x 1
    Xv = torch.matmul(_x, torch.sum(model.v, dim=2, keepdim=True)).permute(1, 0, 2)  # N*d times P*d*1 -> N*P*1
    loss = loss + rho * torch.sum(torch.nn.functional.relu(torch.mul(1 - 2 * sign_patterns, Xv)))
    Xw = torch.matmul(_x, torch.sum(model.w, dim=2, keepdim=True)).permute(1, 0, 2)
    loss = loss + rho * torch.sum(torch.nn.functional.relu(torch.mul(1 - 2 * sign_patterns, Xw)))
    return loss


def validation_cvxproblem(model, testloader, u_vectors, beta, rho, test_len):
    test_loss = 0
    test_correct = 0
    test_noncvx_cost = 0

    model.to(device)
    model.eval()
    with torch.no_grad():
        for _x, _y in testloader:
            _x = _x.to(device)
            _y = _y.to(device)
            _x = _x.view(_x.shape[0], -1)
            _z = (torch.matmul(_x, torch.from_numpy(u_vectors).float().to(device)) >= 0)
            yhat = model(_x, _z).float()
            loss = loss_func_cvxproblem(yhat, one_hot(_y).to(device), model, _x, _z, beta, rho)
            test_loss += loss.cpu().detach().numpy()
            test_correct += torch.eq(torch.argmax(yhat, dim=1), _y).float().sum()
            test_noncvx_cost += get_nonconvex_cost(one_hot(_y).to(device), model, _x, beta)

    return test_loss / test_len, test_correct / test_len, test_noncvx_cost / test_len


def sgd_solver_cvxproblem(ds_train, ds_test, num_epochs, num_neurons, beta, lr, batch_size, rho, u_vectors,
                          solver_type, LBFGS_param, train_len=50000, d=3072, num_classes=10, tst_len=10000):
    # create the model
    model = CustomCVXLayer(d, num_neurons, num_classes).to(device)

    if solver_type == "sgd":
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    elif solver_type == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)  # ,
    elif solver_type == "adagrad":
        optimizer = torch.optim.Adagrad(model.parameters(), lr=lr)  # ,
    elif solver_type == "adadelta":
        optimizer = torch.optim.Adadelta(model.parameters(), lr=lr)  # ,
    elif solver_type == "LBFGS":
        optimizer = torch.optim.LBFGS(model.parameters(), history_size=LBFGS_param[0], max_iter=LBFGS_param[1])  # ,
    else:
        raise ValueError(f'Invalid solver_type: {solver_type}')
    # arrays for saving the loss and accuracy 
    losses = np.zeros(num_epochs + 1)
    accs = np.zeros_like(losses)
    noncvx_losses = np.zeros_like(losses)

    losses_test = np.zeros_like(losses)
    accs_test = np.zeros_like(losses_test)
    noncvx_losses_test = np.zeros_like(losses_test)

    times = np.zeros_like(losses_test)
    times[0] = time.time()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, verbose=True, factor=0.5, eps=1e-12)

    # loss on the entire test set
    losses_test[0], accs_test[0], noncvx_losses_test[0] = \
        validation_cvxproblem(model, ds_test, u_vectors, beta, rho, tst_len)

    print('starting training')
    for i in tqdm(range(1, num_epochs + 1)):
        model.to(device)
        model.train()
        for _x, _y, _z in ds_train:
            optimizer.zero_grad()  # zero the gradients on each pass before the update
            # =========make input differentiable=======================
            _x = _x.float().to(device)
            _y = _y.float().to(device)
            _z = _z.float().to(device)

            # ========forward pass=====================================
            yhat = model(_x, _z).float()

            loss = loss_func_cvxproblem(yhat, one_hot(_y).to(device), model, _x, _z, beta, rho) / len(_y)
            # =======backward pass=====================================
            loss.backward()  # backpropagate the loss through the model
            optimizer.step()  # update the gradients w.r.t the loss

            losses[i] += loss.cpu().detach().numpy() * len(_y)  # loss on the minibatch
            accs[i] += torch.eq(torch.argmax(yhat, dim=1), _y).float().sum()
            noncvx_losses[i] += get_nonconvex_cost(one_hot(_y).to(device), model, _x, beta)

        times[i] = time.time()
        losses[i] /= train_len
        accs[i] /= train_len
        noncvx_losses[i] /= train_len
        print(f"Epoch [{i}/{num_epochs}], "
              f"TRAIN: noncvx/cvx loss: {noncvx_losses[i]:.3f}, {losses[i]:.3f} acc: {accs[i]:.3f}.")
        # get test loss and accuracy
        losses_test[i], accs_test[i], noncvx_losses_test[i] = validation_cvxproblem(
            model, ds_test, u_vectors, beta, rho, tst_len
        )  # loss on the entire test set
        scheduler.step(losses[i])
        print(f"TEST: noncvx/cvx loss: {noncvx_losses_test[i]:.3f}, {losses_test[i]:.3f} acc: {accs_test[i]:.3f}")
    times = times - times[0]
    return noncvx_losses, accs, noncvx_losses_test, accs_test, times, losses, losses_test


class Args:
    GD = False
    CVX = True
    seed = 42
    n_epochs = (100, 100)
    solver_cvx = 'sgd'  # pick: "sgd", "adam", "adagrad", "adadelta", "LBFGS"
    # device = 'cuda' if torch.cuda.is_available() else 'cpu'
    batch_size = 10000
    device = 'cpu'
    num_neurons = 4096  # number of neurons is equal to number of hyperplane arrangements


if __name__ == '__main__':
    matplotlib.rcParams.update({
        'pdf.fonttype': 42,
        'ps.fonttype': 42,
        'lines.linewidth': 3.0,
        'font.size': 24,
        'legend.fontsize': 15,
        'figure.figsize': [2 * 6.4, 2 * 4.8],
    })
    # args = parse_args()
    args = Args() # alternative approach to set within script

    if not args.GD and not args.CVX:  # Validation
        raise AssertionError('-skipGD and -skipCVX cannot be specified simultaneously')
    random.seed(a=args.seed)
    np.random.seed(seed=args.seed)
    torch.manual_seed(seed=args.seed)

    directory = Path(__file__).parent
    device = torch.device(args.device)
    start_time = f'{dt.now():%Y-%m-%d_%H-%M}'

    normalize = transforms.Normalize(mean=[0.507, 0.487, 0.441], std=[0.267, 0.256, 0.276])
    train_dataset = datasets.CIFAR10(directory, train=True, download=True,
                                     transform=transforms.Compose([transforms.ToTensor(), normalize]))
    test_dataset = datasets.CIFAR10(directory, train=False, download=True,
                                    transform=transforms.Compose([transforms.ToTensor(), normalize]))

    n, p1, p2, c = train_dataset.data.shape
    d = c * p1 * p2

    # problem parameters
    beta_ = 1e-3  # regularization parameter
    num_epochs1, num_epochs2 = args.n_epochs
    LBFGS_params = [10, 4]  # these parameters are for the LBFGS solver

    # create dataloaders
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    test_len = test_dataset.data.shape[0]

    # SGD solver for the nonconvex problem
    if args.GD:
        solver_type = "sgd"  # pick: "sgd", "adam", "adagrad", "adadelta", "LBFGS"
        schedule = 0  # learning rate schedule (0: Nothing, 1: ReduceLROnPlateau, 2: ExponentialLR)
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, sampler=None)

        ## SGD constant
        results_noncvx_sgd = []
        for learning_rate in [1e-2, 5e-3, 1e-3]:
            print('SGD-training-mu={}'.format(learning_rate))
            results_noncvx_sgd.append(sgd_solver_pytorch_v2(
                train_loader, test_loader, num_epochs1, args.num_neurons, beta_, learning_rate, args.batch_size,
                solver_type, schedule, LBFGS_params, num_classes=10, d_in=d, train_len=n, test_len=test_len
            ))

        print('Saving GD results')
        torch.save([num_epochs1, [x[:5] for x in results_noncvx_sgd]], f'results_gd_{start_time}.pt')

    # Solver for the convex problem
    if args.CVX:
        rho = 1e-2  # coefficient to penalize the violated constraints
        #  Convex
        print('Generating sign patterns')
        A, y_ = next(iter(DataLoader(train_dataset, batch_size=n, shuffle=False)))
        A = A.view(n, -1)
        sign_pattern_list, u_vector_list = generate_sign_patterns(A, args.num_neurons)
        sign_patterns = np.array([sign_pattern_list[i].int().data.numpy() for i in range(args.num_neurons)])
        u_vectors = np.asarray(u_vector_list).reshape((args.num_neurons, A.shape[1])).T

        train_dataset = PrepareData3D(x=A, y=y_, z=sign_patterns.T)
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

        results_cvx = []
        for learning_rate in [1e-6, 5e-7]:
            print(f'Convex Random-mu={learning_rate}')
            results_cvx.append(sgd_solver_cvxproblem(
                train_loader, test_loader, num_epochs2, args.num_neurons, beta_, learning_rate, args.batch_size, rho,
                u_vectors,
                args.solver_cvx, LBFGS_params, train_len=n, tst_len=test_len
            ))

        #  Convex with convolutional patterns
        print('Generating conv sign patterns')
        sign_pattern_list, u_vector_list = generate_conv_sign_patterns(A, args.num_neurons, c, p1, p2)
        sign_patterns = np.array([sign_pattern_list[i].int().data.numpy() for i in range(args.num_neurons)])
        u_vectors = np.asarray(u_vector_list).reshape((args.num_neurons, A.shape[1])).T

        train_dataset = PrepareData3D(x=A, y=y_, z=sign_patterns.T)
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

        #  Convex Conv1
        results_cvx_conv = []
        for learning_rate in [1e-6, 5e-7]:
            print('Convex Conv1-mu={}'.format(learning_rate))
            results_cvx_conv.append(sgd_solver_cvxproblem(
                train_loader, test_loader, num_epochs2, args.num_neurons, beta_, learning_rate, args.batch_size, rho,
                u_vectors, args.solver_cvx, LBFGS_params, train_len=n, tst_len=test_len
            ))

        print('Saving CVX results')
        torch.save([num_epochs2, results_cvx, results_cvx_conv], f'results_cvx_{start_time}.pt')

    if args.GD and args.CVX:
        # generate and save plots
        mark_sgd = 10
        mark_cvx = 30
        marker_size_sgd = 10
        marker_size_cvx = 12

        plt.figure()  # To plot results in the validation set
        plot_no = 3  # select --> 2: cost, 3: accuracy
        plt.plot(results_noncvx_sgd[0][4], results_noncvx_sgd[0][plot_no][1:], '--', color='darkred',
                 markevery=mark_sgd, markersize=marker_size_sgd, label=r"SGD $\mu=1e-2$")
        plt.plot(results_noncvx_sgd[1][4], results_noncvx_sgd[1][plot_no][1:], '--', color='red',
                 markevery=mark_sgd, markersize=marker_size_sgd, label=r"SGD $\mu=5e-3$")
        plt.plot(results_noncvx_sgd[2][4], results_noncvx_sgd[2][plot_no][1:], '--', color='lightcoral',
                 markevery=mark_sgd, markersize=marker_size_sgd, label=r"SGD $\mu=1e-3$")

        plt.plot(results_cvx[0][4] - results_cvx[0][4][0], results_cvx[0][plot_no], 'o--', color='g',
                 markevery=mark_cvx, markersize=marker_size_cvx, label=r"Convex Random $\mu=1e-6$")
        plt.plot(results_cvx[1][4] - results_cvx[1][4][0], results_cvx[1][plot_no], 'o--', color='lime',
                 markevery=mark_cvx, markersize=marker_size_cvx, label=r"Convex Random $\mu=5e-7$")

        plt.plot(results_cvx_conv[0][4] - results_cvx_conv[0][4][0], results_cvx_conv[0][plot_no], 'o--', color='b',
                 markevery=mark_cvx, markersize=marker_size_cvx, label=r"Convex Conv $\mu=1e-6$")
        plt.plot(results_cvx_conv[1][4] - results_cvx_conv[1][4][0], results_cvx_conv[1][plot_no], 'o--', color='lightblue',
                 markevery=mark_cvx, markersize=marker_size_cvx, label=r"Convex Conv $\mu=5e-7$")

        plt.legend()
        plt.xlabel('Time(s)')
        plt.ylabel("Test Accuracy")
        plt.grid(True, which='both')
        plt.ylim(0.3, 0.6)
        plt.xlim(0, 2500)
        plt.savefig('plots/cifar_multiclass_stepsize_testacc.png', format='png', bbox_inches='tight')

        plt.figure()  # To plot training  acc

        plt.plot(results_noncvx_sgd[0][4], results_noncvx_sgd[0][1], '-', color='darkred',
                 markevery=mark_sgd, markersize=marker_size_sgd, label=r"SGD $\mu=1e-2$")
        plt.plot(results_noncvx_sgd[1][4], results_noncvx_sgd[1][1], '-', color='red',
                 markevery=mark_sgd, markersize=marker_size_sgd, label=r"SGD $\mu=5e-2$")
        plt.plot(results_noncvx_sgd[2][4], results_noncvx_sgd[2][1], '-', color='lightcoral',
                 markevery=mark_sgd, markersize=marker_size_sgd, label=r"SGD $\mu=1e-3$")

        plt.plot(results_cvx[0][4] - results_cvx[0][4][0], results_cvx[0][1], 'o-', color='g',
                 markevery=mark_cvx, markersize=marker_size_cvx, label=r"Convex Random $\mu=1e-6$")
        plt.plot(results_cvx[1][4] - results_cvx[1][4][0], results_cvx[1][1], 'o-', color='lime',
                 markevery=mark_cvx, markersize=marker_size_cvx, label=r"Convex Random $\mu=5e-7$")
        plt.plot(results_cvx_conv[0][4] - results_cvx_conv[0][4][0], results_cvx_conv[0][1], 'o-', color='b',
                 markevery=mark_cvx, markersize=marker_size_cvx, label=r"Convex Conv $\mu=1e-6$")
        plt.plot(results_cvx_conv[1][4] - results_cvx_conv[1][4][0], results_cvx_conv[1][1], 'o-', color='lightblue',
                 markevery=mark_cvx, markersize=marker_size_cvx, label=r"Convex Conv $\mu=5e-7$")

        plt.xlim(0, 2500)
        plt.xlabel('Time(s)')
        plt.ylabel("Training Accuracy")
        plt.grid(True, which='both')
        plt.legend()
        matplotlib.pyplot.grid(True, which="both")
        plt.savefig('plots/cifar_multiclass_stepsize_tracc.png', format='png', bbox_inches='tight')

        # Plot training loss
        plt.figure()
        plt.semilogy(results_noncvx_sgd[0][4], results_noncvx_sgd[0][0], '-', color='darkred',
                     markevery=mark_sgd, markersize=marker_size_sgd, label=r"SGD $\mu=1e-2$")
        plt.semilogy(results_noncvx_sgd[1][4], results_noncvx_sgd[1][0], '-', color='red',
                     markevery=mark_sgd, markersize=marker_size_sgd, label=r"SGD $\mu=5e-2$")
        plt.semilogy(results_noncvx_sgd[2][4], results_noncvx_sgd[2][0], '-', color='lightcoral',
                     markevery=mark_sgd, markersize=marker_size_sgd, label=r"SGD $\mu=1e-3$")

        plt.semilogy(results_cvx[0][4], results_cvx[0][5], 'o-', color='g',
                     markevery=mark_cvx, markersize=marker_size_cvx, label=r"Convex Random $\mu=1e-6$")
        plt.semilogy(results_cvx[1][4], results_cvx[1][5], 'o-', color='lime',
                     markevery=mark_cvx, markersize=marker_size_cvx, label=r"Convex Random $\mu=5e-7$")
        plt.semilogy(results_cvx_conv[0][4], results_cvx_conv[0][5], 'o-', color='b',
                     markevery=mark_cvx, markersize=marker_size_cvx, label=r"Convex Conv $\mu=1e-6$")
        plt.semilogy(results_cvx_conv[1][4], results_cvx_conv[1][5], 'o-', color='lightblue',
                     markevery=mark_cvx, markersize=marker_size_cvx, label=r"Convex Conv $\mu=5e-7$")

        plt.xlim(0, 2500)
        plt.xlabel('Time(s)')
        plt.ylabel("Objective Value")
        plt.grid(True, which="both")
        plt.legend()
        plt.savefig('plots/cifar_multiclass_stepsize_obj.png', format='png', bbox_inches='tight')
