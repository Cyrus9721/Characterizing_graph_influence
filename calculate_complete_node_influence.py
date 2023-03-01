import numpy as np
import pandas as pd
import torch
import os
from dgl import function as fn
from dataset import load_graph_dataset
from model_softmax import SimplifiedGraphNeuralNetwork, fast_hess, fast_hess_cuda, fast_get_inv_hvp_cuda
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import log_loss
from tqdm import tqdm
import cupy as cp
import argparse
import matplotlib.pyplot as plt

def parse_args():
    parser = argparse.ArgumentParser(description='swtich dataset name')
    parser.add_argument('--dataname', type=str, default='cora',
                        help='dataname to be tunned')
    parser.add_argument('--l2_reg', type=float, default=0.1,
                        help='l2_regularization (default: 0.1)')
    args = parser.parse_args()
    return args
