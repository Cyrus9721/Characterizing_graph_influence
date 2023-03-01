import argparse, time, math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl.function as fn
import dgl
from dgl.data import register_data_args
from dgl.data import CoraGraphDataset, CiteseerGraphDataset, PubmedGraphDataset
from dgl.nn.pytorch.conv import SGConv
from dataset import load_graph_dataset


def evaluate(model, g, features, labels, mask):
    model.eval()
    with torch.no_grad():
        logits = model(g, features)[mask]  # only compute the evaluation set
        labels = labels[mask]
        _, indices = torch.max(logits, dim=1)
        correct = torch.sum(indices == labels)
        return correct.item() * 1.0 / len(labels)


class sgc_with_edge_removal:

    def __init__(self, dataset, remove_edge_index=False, lr=1e-2, n_epochs=200, n_hidden=16, n_layers=1,
                 weight_decay=5e-4, dropout=0):
        self.dataset = dataset
        self.remove_edge_idx = remove_edge_index
        # self.gpu = gpu
        self.lr = lr
        self.n_epochs = n_epochs
        self.n_hidden = n_hidden
        self.n_layers = n_layers
        self.weight_decay = weight_decay
        self.dropout = dropout

    def train_evaluate(self):

        g, features, labels, train_mask, val_mask, test_mask, n_classes = load_graph_dataset(self.dataset)

        # if self.gpu < 0:
        #     cuda = False
        # else:
        #     cuda = True

        if self.remove_edge_idx:
            g = dgl.remove_edges(g, self.remove_edge_idx)

        in_feats = features.shape[1]
        n_edges = g.number_of_edges()
        # normalization
        degs = g.in_degrees().float()
        norm = torch.pow(degs, -0.5)
        norm[torch.isinf(norm)] = 0

        # if cuda:
        #     norm = norm.cuda()

        g.ndata['norm'] = norm.unsqueeze(1)

        # create GCN model
        model = SGConv(in_feats, n_classes, k=2, allow_zero_in_degree=True)

        # if cuda:
        #     model.cuda()

        loss_fcn = torch.nn.CrossEntropyLoss()

        # use optimizer
        optimizer = torch.optim.Adam(model.parameters(),
                                     lr=self.lr,
                                     weight_decay=self.weight_decay)

        # initialize graph
        dur = []
        for epoch in range(self.n_epochs):
            model.train()
            if epoch >= 3:
                t0 = time.time()
            # forward
            logits = model(g, features)
            loss = loss_fcn(logits[train_mask], labels[train_mask])

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if epoch >= 3:
                dur.append(time.time() - t0)

            acc = evaluate(model, g, features, labels, val_mask)
            print("Epoch {:05d} | Time(s) {:.4f} | Loss {:.4f} | Accuracy {:.4f} | "
                  "number of edges {:.2f}".format(epoch, np.mean(dur), loss.item(),
                                                  acc, n_edges))

        print()
        acc = evaluate(model, g, features, labels, test_mask)
        print("Test accuracy {:.2%}".format(acc))

        return acc


# t = sgc_with_edge_removal(dataset='cora')
# t.train_evaluate()
