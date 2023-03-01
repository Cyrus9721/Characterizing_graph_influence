import argparse
import time
import numpy as np
import torch
import torch.nn.functional as F
import dgl
from dataset import load_graph_dataset
from gcn import GCN


def evaluate(model, features, labels, mask):
    model.eval()
    with torch.no_grad():
        logits = model(features)
        logits = logits[mask]
        labels = labels[mask]
        _, indices = torch.max(logits, dim=1)
        correct = torch.sum(indices == labels)
        return correct.item() * 1.0 / len(labels)


class gcn_with_node_flipping:

    def __init__(self, graph, features, new_labels, train_mask, val_mask, test_mask, num_classes,
                 lr=1e-2, n_epochs=200, n_hidden=16, n_layers=1, weight_decay=5e-4, dropout=0):

        # self.gpu = gpu
        self.graph = graph
        self.features = features
        self.new_labels = new_labels

        self.train_mask = train_mask
        self.val_mask = val_mask
        self.test_mask = test_mask
        self.num_classes = num_classes

        self.lr = lr
        self.n_epochs = n_epochs
        self.n_hidden = n_hidden
        self.n_layers = n_layers
        self.weight_decay = weight_decay
        self.dropout = dropout

    def train_evaluate(self):

        g = self.graph
        features = self.features
        labels = self.new_labels
        train_mask = self.train_mask
        val_mask = self.val_mask
        test_mask = self.test_mask
        n_classes = self.num_classes
        # if self.gpu < 0:
        #     cuda = False
        # else:
        #     cuda = True

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
        model = GCN(g,
                    in_feats,
                    self.n_hidden,
                    n_classes,
                    self.n_layers,
                    F.relu,
                    self.dropout)

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
            logits = model(features)
            loss = loss_fcn(logits[train_mask], labels[train_mask])

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if epoch >= 3:
                dur.append(time.time() - t0)

            acc = evaluate(model, features, labels, val_mask)
            # print("Epoch {:05d} | Time(s) {:.4f} | Loss {:.4f} | Accuracy {:.4f} | "
            #       "number of edges {:.2f}". format(epoch, np.mean(dur), loss.item(),
            #                                      acc, n_edges))

        acc = evaluate(model, features, labels, test_mask)
        print("Test accuracy {:.2%}".format(acc))

        return acc
