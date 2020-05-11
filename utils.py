import torch
import numpy as np
from typing import Union, List
from warnings import warn


def check_xy(x: torch.Tensor, y: torch.Tensor):
    """check dimension of x and y"""
    assert len(x.shape) <= 2, "x has too many dimensions"
    assert len(x.shape) > 0, "x can not be empty"
    if len(x.shape) == 1:
        x = x.reshape(-1, 1)
    assert len(y.shape) <= 2, "y has too many dimensions"
    assert len(y.shape) > 0, "y can not be empty"
    assert y.shape[0] == x.shape[0], "x and y must have the same sample size"
    if len(y.shape) == 2:
        y = y.squeeze()
    return x, y


def sigmoid(x: torch.Tensor):
    """sigmoid function defined as exp(x)/[1+exp(x)]"""
    return torch.exp(x) / (1 + torch.exp(x))


def numpy_to_torch(x: Union[torch.Tensor, np.ndarray]):
    """transform numpy.ndarray to torch.Tensor"""
    if isinstance(x, np.ndarray):
        x = torch.from_numpy(x)
    return x


def add_intercept(x: torch.Tensor, group_size: List[int] = None):
    """add intercept to data matrix and group size"""
    if all(x[:, 0] == 1):
        if group_size is None:
            warn("Looks like intercept is already included, nothing has been added.")
            return x
        elif group_size[0] == 1:
            warn("Looks like intercept is already included, nothing has been added.")
            return x, group_size
        else:
            x = torch.cat((torch.ones([x.shape[0], 1]).float(), x.float()), 1)
            group_size = [1] + group_size
            return x, group_size
    else:
        x = torch.cat((torch.ones([x.shape[0], 1]).float(), x.float()), 1)
        if group_size is not None:
            group_size = [1] + group_size
            return x, group_size
        else:
            return x

def remove_intercept(x: torch.Tensor):
    """removes intercept if it's already there"""
    if all(x[:, 0] == 1):
        x = x[:, 1:]
    return x

def compute_nonzeros(beta: torch.Tensor, group_size: List[int]):
    """computes how many nonzero groups"""
    num = 0
    start = 0
    for i in range(len(group_size)):
        if torch.norm(beta[start: start + group_size[i]]) != 0:
            num += 1
        start += group_size[i]
    return num