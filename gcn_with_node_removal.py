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


def remove_nodes(graph, train_mask, node_id_list):
    # node_id_list: node id in the training graph
    g = graph.clone()
    new_train_mask = train_mask.clone()

    from_indexes, to_indexes = g.edges()
    all_edge_remove_list = []
    for node_id in node_id_list:
        node_from_indexes = torch.where(from_indexes == node_id)[0]
        node_to_indexes = torch.where(to_indexes == node_id)[0]
        node_id_indexes = torch.unique(torch.concat([node_from_indexes, node_to_indexes]))
        node_id_indexes = list(node_id_indexes.numpy())
        all_edge_remove_list.extend(node_id_indexes)

    all_edge_remove_list = torch.tensor(np.unique(all_edge_remove_list))
    g.remove_edges(all_edge_remove_list)

    new_train_mask[node_id_list] = False
    return g, new_train_mask


class gcn_with_node_list_removal:

    def __init__(self, dataset, remove_node_list=[], lr=1e-2, n_epochs=200, n_hidden=16, n_layers=1, weight_decay=5e-4,
                 dropout=0.5):
        self.dataset = dataset
        self.remove_node_list = remove_node_list
        # self.gpu = gpu
        self.lr = lr
        self.n_epochs = n_epochs
        self.n_hidden = n_hidden
        self.n_layers = n_layers
        self.weight_decay = weight_decay
        self.dropout = dropout

    def train_evaluate(self):

        g_orig, features, labels, train_mask_orig, val_mask, test_mask, n_classes = load_graph_dataset(self.dataset)

        g, train_mask = remove_nodes(graph=g_orig, train_mask=train_mask_orig, node_id_list=self.remove_node_list)

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
        #     print("Epoch {:05d} | Time(s) {:.4f} | Loss {:.4f} | Accuracy {:.4f} | "
        #           "number of edges {:.2f}".format(epoch, np.mean(dur), loss.item(),
        #                                           acc, n_edges))
        #
        # print()
        acc = evaluate(model, features, labels, test_mask)
        # print("Test accuracy {:.2%}".format(acc))

        return acc
