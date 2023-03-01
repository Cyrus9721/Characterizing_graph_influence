import numpy as np
import pandas as pd
import torch
from dgl import function as fn
from sklearn.preprocessing import OneHotEncoder
from dataset import load_graph_dataset
from tqdm import tqdm
from model_softmax import SimplifiedGraphNeuralNetwork
from sklearn.linear_model import LogisticRegression


class train_preprocessed_data:
    def __init__(self, data_name, k):
        self.data_name = data_name
        self.k = k

    def load_graph_to_tabular(self):
        graph, feat, labels, train_mask, val_mask, test_mask, number_classes = load_graph_dataset(self.data_name)
        feat0 = feat.clone()
        degs = graph.in_degrees().float().clamp(min=1)
        norm = torch.pow(degs, -0.5)
        norm = norm.to(feat0.device).unsqueeze(1)

        for _ in range(self.k):
            feat0 = feat0 * norm
            graph.ndata['h'] = feat0
            graph.update_all(fn.copy_u('h', 'm'),
                             fn.sum('m', 'h'))
            feat0 = graph.ndata.pop('h')
            feat0 = feat0 * norm

        train_x = feat0[train_mask].numpy().astype(np.float64)
        train_y = labels[train_mask].numpy().astype(np.float64)

        val_x = feat0[val_mask].numpy().astype(np.float64)
        val_y = labels[val_mask].numpy().astype(np.float64)

        test_x = feat0[test_mask].numpy().astype(np.float64)
        test_y = labels[test_mask].numpy().astype(np.float64)

        return train_x, train_y, test_x, test_y, val_x, val_y
