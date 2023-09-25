import scipy as sp
import torch
import torch.nn as nn
import numpy as np
import utils.batch_linalg as linalg
from typing import Optional


def CalculateOutSize(blocks, channels, samples):
    '''
    Calculate the output based on input size.
    model is from nn.Module and inputSize is a array.
    '''
    x = torch.rand(1, 1, channels, samples)
    for block in blocks:
        block.eval()
        x = block(x)
    x = x.reshape(x.size(0), -1)
    return x.shape[-1]
    

def ledoit_wolf(X, assume_centered=False):
    """Estimates the shrunk Ledoit-Wolf covariance matrix.

    Read more in the :ref:`User Guide <shrunk_covariance>`.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        Data from which to compute the covariance estimate

    assume_centered : bool, default=False
        If True, data will not be centered before computation.
        Useful to work with data whose mean is significantly equal to
        zero but is not exactly zero.
        If False, data will be centered before computation.

    Returns
    -------
    shrunk_cov : ndarray of shape (n_features, n_features)
        Shrunk covariance.

    Notes
    -----
    The regularized (shrunk) covariance is:

    (1 - shrinkage) * cov + shrinkage * mu * np.identity(n_features)

    where mu = trace(cov) / n_features
    """

    if not assume_centered:
        X = X - torch.mean(X, dim=1, keepdim=True)
    _, n_samples, n_features = X.shape

    X2 = X ** 2
    emp_cov_trace = torch.sum(X2, dim=1) / n_samples
    mu = torch.sum(emp_cov_trace, dim=1) / n_features
    beta_ = 0.0
    delta_ = 0.0
    delta_ += torch.sum(torch.matmul(torch.transpose(X, 2, 1), X) ** 2, dim=(1, 2))
    delta_ /= n_samples ** 2
    beta_ += torch.sum(torch.matmul(torch.transpose(X2, 2, 1), X2), dim=(1, 2))
    beta = 1.0 / (n_features * n_samples) * (beta_ / n_samples - delta_)
    delta = delta_ - 2.0 * mu * emp_cov_trace.sum(1) + n_features * mu ** 2
    delta /= n_features
    beta = torch.cat([beta.reshape(-1, 1), delta.reshape(-1, 1)], dim=1)
    beta, _ = torch.min(beta, dim=1)
    shrinkage = beta / delta

    X_ = torch.transpose(X, 2, 1)
    X_ = X_ - torch.mean(X_, dim=-1, keepdim=True)
    emp_cov = torch.matmul(X_, torch.transpose(X_, 2, 1)) / (1.0 * X_.shape[2])
    mu = torch.einsum('bii->b', emp_cov) / n_features
    shrunk_cov = (1.0 - shrinkage.reshape(-1, 1, 1)) * emp_cov
    shrunk_cov.flatten(start_dim=1, end_dim=2)[:, :: n_features + 1] += (shrinkage * mu).reshape(-1, 1)

    return shrunk_cov


def oas(X, *, assume_centered=False):
    """Estimate covariance with the Oracle Approximating Shrinkage algorithm.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        Data from which to compute the covariance estimate.

    assume_centered : bool, default=False
      If True, data will not be centered before computation.
      Useful to work with data whose mean is significantly equal to
      zero but is not exactly zero.
      If False, data will be centered before computation.

    Returns
    -------
    shrunk_cov : array-like of shape (n_features, n_features)
        Shrunk covariance.

    shrinkage : float
        Coefficient in the convex combination used for the computation
        of the shrunk estimate.

    Notes
    -----
    The regularised (shrunk) covariance is:

    (1 - shrinkage) * cov + shrinkage * mu * np.identity(n_features)

    where mu = trace(cov) / n_features

    The formula we used to implement the OAS is slightly modified compared
    to the one given in the article. See :class:`OAS` for more details.
    """
    
    _, n_samples, n_features = X.shape

    X_ = torch.transpose(X, 2, 1)
    if not assume_centered:
        X_ = X_ - torch.mean(X_, dim=-1, keepdim=True)
    emp_cov = torch.matmul(X_, torch.transpose(X_, 2, 1)) / (1.0 * X_.shape[2])
    mu = linalg.trace(emp_cov) / n_features

    # formula from Chen et al.'s **implementation**
    alpha = torch.mean(emp_cov ** 2, dim=(1, 2))
    num = alpha + mu ** 2
    den = (n_samples + 1.0) * (alpha - (mu ** 2) / n_features)
    shrinkage = torch.cat([(num / den).reshape(-1, 1), torch.ones_like(den.reshape(-1, 1)).to(X.device)], dim=1)
    shrinkage, _ = torch.min(shrinkage, dim=1)
    shrunk_cov = (1.0 - shrinkage.reshape(-1, 1, 1)) * emp_cov
    shrunk_cov.flatten(start_dim=1, end_dim=2)[:, :: n_features + 1] += (shrinkage * mu).reshape(-1, 1)

    return shrunk_cov, shrinkage


class xDAWNCovariances(nn.Module):
    def __init__(self, filters, Ps):
        super(xDAWNCovariances, self).__init__()
        self.nb_filters = int(filters.shape[0]/2)
        self.Ps = Ps
        self.xdawn_filters = nn.Parameter(filters)
    
    def forward(self, x):
        X_ = []
        for i in range(len(self.Ps)):
            P = self.Ps[i]
            temp_P = torch.matmul(self.xdawn_filters[i*self.nb_filters:(i+1)*self.nb_filters, :], P)
            temp_P = temp_P.unsqueeze(0).tile((x.shape[0], 1, 1))
            X_.append(temp_P)
        kernel = self.xdawn_filters.unsqueeze(0)
        X_.append(torch.matmul(kernel, x))
        X_ = torch.cat(X_, axis=1)
        # convariance
        # xDAWN_cov = torch.zeros((X_.shape[0], X_.shape[1], X_.shape[1])).to(x.device)
        # xDAWN_cov = ledoit_wolf(torch.transpose(X_, 2, 1))
        xDAWN_cov, _ = oas(torch.transpose(X_, 2, 1))

        return xDAWN_cov


def tangent_space(covmats, Cref, epsilon):
    """Project a set of covariance matrices in the tangent space. according to
    the reference point Cref

    :param covmats: np.ndarray
        Covariance matrices set, Ntrials X Nchannels X Nchannels
    :param Cref: np.ndarray
        The reference covariance matrix
    :returns: np.ndarray
        the Tangent space , a matrix of Ntrials X (Nchannels*(Nchannels+1)/2)

    """
    Nt, Nc, Nc = covmats.shape
    Cm12 = linalg.sym_inv_sqrtm1(Cref)
    Cm12 = Cm12.unsqueeze(dim=0)
    m = torch.matmul(torch.matmul(Cm12, covmats), Cm12) + torch.diag(epsilon * torch.randn(Nc)).to(covmats.device)
    m = linalg.sym_logm(m)
    coeffs = (np.sqrt(2) * np.triu(np.ones((Nc, Nc)), 1) + np.eye(Nc))
    coeffs = torch.from_numpy(coeffs).type(covmats.dtype).to(covmats.device)
    coeffs = coeffs.unsqueeze(0)

    m = torch.multiply(m, coeffs)
    T_list = []
    for i in range(Nc):
        T_list.append(torch.reshape(m[:, i, i:], shape=(Nt, -1)))
    T = torch.cat(T_list, axis=1)
    return T



class TangentSpace(nn.Module):
    def __init__(self, Cref, epsilon=1e-6):
        super(TangentSpace, self).__init__()
        self.epsilon = epsilon
        self.dim = Cref.shape[-1]
        self.source = linalg.sym_sqrtm(Cref)
        self.sqrt_cov = nn.Parameter(self.source.type(torch.FloatTensor))
        # self.Cref = nn.Parameter(Cref.type(torch.FloatTensor))

    def forward(self, x):
        Cref_tensor = torch.matmul(self.sqrt_cov, torch.transpose(self.sqrt_cov, 1, 0)) + self.epsilon * torch.eye(self.dim).to(x.device)
        return tangent_space(x, Cref_tensor, self.epsilon)


class CSP(nn.Module):
    def __init__(self, filters, transform_into='csp_space', log=False):
        """
        CSP Layer
        :param filters: CSP spatial filters, (filters, channels).
        :param transform_into: 'csp_space' or 'average_power'.
        :param log: only needed when transform_into=='average_power'
        :param kwargs:
        """
        super(CSP, self).__init__()
        self.nb_filters = int(filters.shape[0])
        self.transform_into = transform_into
        self.log = log 
        self.csp_filters = nn.Parameter(filters)

    def forward(self, x):
        kernel = self.csp_filters.unsqueeze(0).tile((x.shape[0], 1, 1))
        x_ = torch.matmul(kernel, x)
        
        if self.transform_into == 'csp_space':
            return x_
        elif self.transform_into == 'average_power':
            power = torch.mean(torch.square(x_), dim=2, keepdim=False)
            if self.log: power = torch.log(power)
            return power
        else:
            raise Exception(f'{self.transform_into} is not valid!')


class BN(nn.Module):
    def __init__(self, num_features, mean, var):
        super(BN, self).__init__()
        self.bn = nn.BatchNorm1d(num_features=num_features)
        self.bn.running_mean = mean
        self.bn.running_var = var
    
    def forward(self, x):
        return self.bn(x)


class LogisticRegression(nn.Module):
    def __init__(self, in_features, out_features, weight, bias):
        super(LogisticRegression, self).__init__()
        self.linear = nn.Linear(in_features=in_features, out_features=out_features)
        self.linear.weight = nn.Parameter(weight)
        self.linear.bias = nn.Parameter(bias)
    def forward(self, x):
        return self.linear(x)
    