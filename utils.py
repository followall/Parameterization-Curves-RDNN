import argparse
import torch
from scipy.special import binom
from torch.linalg import solve, lstsq


def setup_argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--degree', '-d', type=int, dest='d', default=3)
    parser.add_argument('--training-rate', type=int, dest='training_rate', default=0.7)
    parser.add_argument('--batch-size', '-b', type=int, dest='batch_size', default=128)
    parser.add_argument('--epochs', '-e', type=int, dest='epochs', default=500)
    parser.add_argument('--lr', '-l', type=float, dest='lr', default=1e-3)
    parser.add_argument('--seed', '-s', type=int, dest='seed', default=13)
    parser.add_argument('--device', type=str, dest='device', default='cuda')
    return parser


def compute_lambda_from_t(t : torch.tensor) -> torch.tensor:
    """
    params: t (bs, 2*d+1)
    return: lambdas (bs, 2*d-1, 2)
    """
    assert t.size(-1) % 2 == 1
    bs = t.size(0)
    d = t.size(-1) // 2
    lambdas = torch.tensor([])
    for i in range(bs):
        temp = []
        for j in range(1,2*d):
            ld = (t[i][j+1]-t[i][j])/(t[i][j+1]-t[i][j-1])
            temp.append(ld)
            temp.append(1-ld)
        lambdas = torch.cat((lambdas, torch.tensor(temp)), dim=0)
    lambdas = lambdas.view(bs, 2*d-1, 2)
    return lambdas


def compute_t_from_lambda(lambdas: torch.tensor) -> torch.tensor:
    """
    params: lambdas (bs, 2*d-1, 2)
    returns: t (bs, 2*d+1)
    """
    
    assert lambdas.size(-2) % 2 == 1
    d = (lambdas.size(-2) + 1) // 2
    n = 2 * d + 1
    bs = lambdas.size(0)
    A = torch.zeros(bs, n, n)
    A[:,] = torch.eye(n)
    for j in range(1, n-1):
        A[:,j,j-1] = -lambdas[:,j-1,0] # lambda_1
        A[:,j,j+1] = -lambdas[:,j-1,1] # lambda_2
    B = torch.zeros(bs, n)
    B[:,-1] = 1

    B.requires_grad = True
    t = lstsq(A, B).solution
    At = A@t.unsqueeze(-1)
    torch.allclose(At, B.unsqueeze(-1).expand_as(At))
    return t


def compute_control_points_from_t_and_p(t : torch.tensor, p: torch.tensor) -> torch.tensor:
    """
    params:  t (bs, 2*d+1)
    params:  p (bs, 2*d+1, 2)
    returns: c (bs, d+1,   2)
    """
    assert t.size(0) == p.size(0)
    bs = t.size(0)
    n = t.size(1)
    d = n // 2
    t = t.squeeze(-1)
    one_min_t = 1 - t
    T1 = torch.zeros(bs, n, d+1)
    T2 = torch.zeros(bs, n, d+1)
    Binom = torch.zeros(d+1)    
    for i in range(d+1):
        T1[:,:,i] = t ** i
        T2[:,:,i] = one_min_t ** (d-i)
        Binom[i] = binom(d, i)
    T = T1 * T2    # (bs, n, d+1)
    A = T * Binom  # (bs, n, d+1)


    # B = p          # (bs, n, 2)
    # c = lstsq(A, B).solution
    # Ac = A@c
    # torch.allclose(Ac, B)

    A_T = A.transpose(1, 2)
    c = torch.inverse(A_T @ A) @ A_T @ p
    
    
    return c



def compute_p_from_control_points_and_t(c : torch.tensor, t : torch.tensor) -> torch.tensor:
    """
    params:  c (bs, d+1,   2)
    params:  t (bs, 2*d+1)
    returns: p (bs, 2*d+1, 2)
    """
    assert(c.size(0) == t.size(0))
    bs = t.size(0)
    n = t.size(1)
    d = c.size(1) - 1
    t = t.squeeze(-1)
    one_min_t = 1 - t
    T1 = torch.zeros(bs, n, d+1)
    T2 = torch.zeros(bs, n, d+1)
    Binom = torch.zeros(d+1)    
    for i in range(d+1):
        T1[:,:,i] = t ** i
        T2[:,:,i] = one_min_t ** (d-i)
        Binom[i] = binom(d, i)
    T = T1 * T2    # (bs, n, d+1)
    A = T * Binom  # (bs, n, d+1)
    p = A @ c
    return p
