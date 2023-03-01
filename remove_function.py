import torch


def remove_preprocessed_feature(DAD, i):
    return torch.cat((DAD[:i], DAD[i + 1:]))


def change_mask_by_index(y, i):
    y[i] = False
    return y


def remove_feature(A, i):
    return None


def remove_edges(g, i):
    return None


def remove_nodes(g, i):
    return None
