from __future__ import print_function
import torch
import numpy as np

def diff_mse(x, y):
    x_vec = x.view(1, -1).squeeze()
    y_vec = y.view(1, -1).squeeze()
    return torch.mean(torch.pow((x_vec - y_vec), 2)).item()


def conv2d_scalar(x_in, conv_weight, conv_bias, device, progress=False):
    # x_in, conv_weight, conv_bias = x_in.to(
    #     device), conv_weight.to(device), conv_bias.to(device)
    N_batch, C_in, S_in, _ = x_in.size()
    C_out, _, K, _ = conv_weight.size()
    S_out = S_in - K + 1
    x_out = torch.zeros(N_batch, C_out, S_out, S_out).to(device)
    for n in range(N_batch):
        if progress and n % 8 == 0:
            print("[conv2d_scalar] Batch num: ", n)
        for c_out in range(C_out):
            for m in range(S_out):
                for l in range(S_out):
                    for c_in in range(C_in):
                        for i in range(K):
                            for j in range(K):
                                x_out[n, c_out, m, l].add_(
                                    x_in[n, c_in, m+i, l+j] * conv_weight[c_out, c_in, i, j])
                    x_out[n, c_out, m, l].add_(conv_bias[c_out])
    return x_out


# def im2col(X, kernel_size, device, stride=1):
#     C_in, S_in, _ = X.size()
#     S_out = (S_in-kernel_size)//stride+1
#     H_col = kernel_size**2
#     W_col = C_in*S_out*S_out

#     X_col = torch.empty(H_col, W_col).to(device)

#     for c_in in range(C_in):
#       for m in range(S_out):
#         for l in range(S_out):
#           i = m*S_out + l
#           X_col[:,i]=X[c_in, m:m+kernel_size, l:l+kernel_size].flatten()

#     return X_col


# def conv2d_vector(x_in, conv_weight, conv_bias, device, progress=False):
#     N_batch, C_in, S_in, _ = x_in.size()
#     C_out, _, K, _ = conv_weight.size()
#     S_out = S_in - K + 1

#     x_out = torch.empty(N_batch, C_out, S_out, S_out).to(device)
#     weights = conv_weight2rows(conv_weight[:, 0])

#     for n in range(N_batch):
#       if progress:
#         print(n)
#       A_col = im2col(x_in[n], 5, device)
#       x_out[n] = (torch.mm(weights, A_col).t() + conv_bias).t().view(C_out, S_out, S_out)
#     return x_out


def im2col(X, kernel_size, device, stride=1):
    """
    This version of im2col is much faster than described in my previous homework. 
    The main trick - is the usage of the fancy indexing to achieve vectorized performance.
    Also it's the only place where I am using numpy in this homework
    Inspired by the following:
    https://github.com/wiseodd/hipsternet/blob/master/hipsternet/im2col.py
    https://fdsmlhn.github.io/2017/11/02/Understanding%20im2col%20implementation%20in%20Python(numpy%20fancy%20indexing)/
    """
    def get_im2col_indices(x_size, K, stride=1):
        N, C, H, W = x_size
        S_out = (H - K)//stride + 1

        i0 = np.repeat(np.arange(K), K)
        i0 = np.tile(i0, C)
        i1 = stride * np.repeat(np.arange(S_out), S_out)
        j0 = np.tile(np.arange(K), K * C)
        j1 = stride * np.tile(np.arange(S_out), S_out)
        i = i0.reshape(-1, 1) + i1.reshape(1, -1)
        j = j0.reshape(-1, 1) + j1.reshape(1, -1)

        k = np.repeat(np.arange(C), K * K).reshape(-1, 1)

        return (k.astype(int), i.astype(int), j.astype(int))

    k, i, j = get_im2col_indices(X.size(), kernel_size, stride)

    N_batch, C_in, S_in, _ = X.size()
    S_out = (S_in-kernel_size)//stride+1
    H_col = kernel_size**2
    W_col = C_in*S_out*S_out

    X_col = torch.empty(N_batch, H_col, W_col).to(device)
    X_col = X[:, k, i, j]
    return X_col


def conv_weight2rows(conv_weight):
    C_out, K, _ = conv_weight.size()
    return conv_weight.view(C_out, K*K)


def conv2d_vector(x_in, conv_weight, conv_bias, device, progress=False):
    N_batch, C_in, S_in, _ = x_in.size()
    C_out, _, K, _ = conv_weight.size()
    S_out = S_in - K + 1

    x_out = torch.empty(N_batch, C_out, S_out, S_out).to(device)
    weights = conv_weight2rows(conv_weight[:, 0])
    A_col = im2col(x_in, K, device)
    x_out = (torch.matmul(weights, A_col).transpose(1, 2) +
             conv_bias).transpose(1, 2).view(N_batch, C_out, S_out, S_out)
    return x_out


def pool2d_scalar(a, device, progress=False):
    # a = a.to(device)
    N_batch, C_in, S_in, _ = a.size()
    C_out = C_in
    S_out = S_in//2
    a_out = torch.zeros(N_batch, C_out, S_out, S_out).to(device)
    for n in range(N_batch):
        if progress and n % 8 == 0:
            print("[pool2d_scalar] Batch num: ", n)
        for c_out in range(C_out):
            for m in range(S_out):
                for l in range(S_out):
                    a_out[n, c_out, m, l] = torch.max(
                        torch.tensor([a[n, c_out, 2*m, 2*l],
                                      a[n, c_out, 2*m, 2*l+1],
                                      a[n, c_out, 2*m+1, 2*l],
                                      a[n, c_out, 2*m+1, 2*l+1]]))
    return a_out


def pool2d_vector(a, device):
    N_batch, C_in, S_in, _ = a.size()
    C_out = C_in
    S_out = S_in//2
    a_out = torch.empty(N_batch, C_out, S_out, S_out).to(device)
    view = a.view(N_batch, C_in, S_in//2, 2, S_in//2, 2)
    a_out = torch.max((torch.max(view, dim=3).values), dim=4).values
    return a_out


def relu_scalar(a, device, progress=False):
    # a = a.to(device)
    N_batch, S_in = a.size()
    a_out = torch.zeros(N_batch, S_in).to(device)
    for n in range(N_batch):
        if progress and n % 8 == 0:
            print("[relu_scalar] Batch num: ", n)
        for i in range(S_in):
            a_out[n, i] = 0 if 0 > a[n, i] else a[n, i]
    return a_out


def relu_vector(a, device):
    a_out = torch.clone(a).to(device)
    a_out[a_out < 0] = 0
    return a_out


def reshape_vector(a, device):
    N_batch, C_in, S_in, _ = a.size()
    S_out = C_in*S_in*S_in
    a_out = torch.empty(N_batch, S_out).to(device)
    a_out = a.view(N_batch, S_out)
    return a_out


def reshape_scalar(a, device, progress=False):
    # a = a.to(device)
    N_batch, C_in, S_in, _ = a.size()
    S_out = C_in*S_in**2
    a_out = torch.zeros(N_batch, S_out).to(device)
    for n in range(N_batch):
        if progress and n % 8 == 0:
            print("[reshape_scalar] Batch num: ", n)
        for c_in in range(C_in):
            for m in range(S_in):
                for l in range(S_in):
                    j = 144*c_in + 12*m + l
                    a_out[n, j] = a[n, c_in, m, l]
    return a_out


def fc_layer_scalar(a, weight, bias, device, progress=False):
    # a, weight, bias = a.to(device), weight.to(device), bias.to(device)
    N_batch, S_in = a.size()
    S_out = bias.size(0)
    a_out = torch.zeros(N_batch, S_out).to(device)
    for n in range(N_batch):
        if progress and n % 8 == 0:
            print("[fc_layer_scalar] Batch num: ", n)
        for j in range(S_out):
            for i in range(S_in):
                a_out[n, j].add_(a[n, i]*weight[j, i])
            a_out[n, j].add_(bias[j])
    return a_out


def fc_layer_vector(a, weight, bias, device):
    # a, weight, bias = a.to(device), weight.to(device), bias.to(device)
    N_batch, В = a.size()
    S_out = bias.size(0)
    a_out = torch.empty(N_batch, S_out).to(device)
    for n in range(N_batch):
        a_out[n] = torch.matmul(a[n], weight.t()) + bias
    return a_out
