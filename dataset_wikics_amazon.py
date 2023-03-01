# This file is modified from https://github.com/pmernyei/wiki-cs-dataset/blob/master/experiments/node_classification/load_graph_data.py
import sys

sys.path.append('..')

import numpy as np
import json
import itertools
import torch
import networkx as nx
import dgl
import os.path
import warnings

DATA_PATH = os.path.join('wiki-cs-dataset', 'dataset', 'data.json')

from dgl.data import AmazonCoBuyComputerDataset, AmazonCoBuyPhotoDataset


class NodeClassificationDataset:
    def __init__(self, graph, features, labels, train_masks, stopping_masks,
                 val_masks, test_mask, n_edges, n_classes, n_feats):
        self.graph = graph
        self.features = features
        self.labels = labels
        self.train_masks = train_masks
        self.stopping_masks = stopping_masks
        self.val_masks = val_masks
        self.test_mask = test_mask
        self.n_edges = n_edges
        self.n_classes = n_classes
        self.n_feats = n_feats


def from_file(filename, self_loop):
    data = json.load(open(filename))
    features = torch.FloatTensor(np.array(data['features']))
    labels = torch.LongTensor(np.array(data['labels']))
    if hasattr(torch, 'BoolTensor'):
        train_masks = [torch.BoolTensor(tr) for tr in data['train_masks']]
        val_masks = [torch.BoolTensor(val) for val in data['val_masks']]
        stopping_masks = [torch.BoolTensor(st) for st in data['stopping_masks']]
        test_mask = torch.BoolTensor(data['test_mask'])
    else:
        train_masks = [torch.ByteTensor(tr) for tr in data['train_masks']]
        val_masks = [torch.ByteTensor(val) for val in data['val_masks']]
        stopping_masks = [torch.ByteTensor(st) for st in data['stopping_masks']]
        test_mask = torch.ByteTensor(data['test_mask'])
    n_feats = features.shape[1]
    n_classes = len(set(data['labels']))

    g = dgl.DGLGraph()
    # g = dgl.graph()
    g.add_nodes(len(data['features']))
    edge_list = list(itertools.chain(*[[(i, nb) for nb in nbs] for i, nbs in enumerate(data['links'])]))
    n_edges = len(edge_list)
    # add edges two lists of nodes: src and dst
    src, dst = tuple(zip(*edge_list))
    g.add_edges(src, dst)
    # edges are directional in DGL; make them bi-directional
    g.add_edges(dst, src)
    if self_loop:
        g = dgl.remove_self_loop(g)
        g = dgl.add_self_loop(g)
    return NodeClassificationDataset(g, features, labels, train_masks, stopping_masks,
                                     val_masks, test_mask, n_edges, n_classes, n_feats)


def load_wikics(gpu=-1, self_loop=True):
    warnings.filterwarnings("ignore")
    data = from_file(DATA_PATH, self_loop)

    print("""----Data statistics------'
      #Edges %d
      #Classes %d
      #Train samples %d
      #Val samples %d
      #Stopping samples %d
      #Test samples %d""" %
          (data.n_edges, data.n_classes,
           data.train_masks[0].int().sum().item(),
           data.val_masks[0].int().sum().item(),
           data.stopping_masks[0].int().sum().item(),
           data.test_mask.int().sum().item()))

    # Preprocess graph
    if gpu < 0:
        cuda = False
    else:
        cuda = True
        torch.cuda.set_device(gpu)
        data.features = data.features.cuda()
        data.labels = data.labels.cuda()
        data.test_mask = data.test_mask.cuda()
        for i in range(len(data.train_masks)):
            data.train_masks[i] = data.train_masks[i].cuda()
        for i in range(len(data.val_masks)):
            data.val_masks[i] = data.val_masks[i].cuda()
        for i in range(len(data.stopping_masks)):
            data.stopping_masks[i] = data.stopping_masks[i].cuda()

    # graph normalization
    degs = data.graph.in_degrees().float()
    norm = torch.pow(degs, -0.5)
    norm[torch.isinf(norm)] = 0
    if cuda:
        norm = norm.cuda()
    data.graph.ndata['norm'] = norm.unsqueeze(1)
    return data.graph, data.features, data.labels, data.train_masks, data.val_masks, data.test_mask, data.n_classes


def load_amazon(dataname='computer', seed = 42):
    def index_to_mask(index, size):
        mask = torch.zeros(size, dtype=torch.bool, device=index.device)
        mask[index] = 1
        return mask

    def random_amazon_splits(data, num_classes, seed=seed):
        # Set random coauthor/co-purchase splits:
        # * 20 * num_classes labels for training
        # * 30 * num_classes labels for validation
        # rest labels for testing
        torch.manual_seed(seed)
        indices = []
        for i in range(num_classes):
            index = (data[0].ndata['label'] == i).nonzero().view(-1)
            index = index[torch.randperm(index.size(0))]
            indices.append(index)

        train_index = torch.cat([i[:20] for i in indices], dim=0)
        val_index = torch.cat([i[20:50] for i in indices], dim=0)

        rest_index = torch.cat([i[50:] for i in indices], dim=0)
        rest_index = rest_index[torch.randperm(rest_index.size(0))]

        train_mask = index_to_mask(train_index, size=data[0].number_of_nodes())
        val_mask = index_to_mask(val_index, size=data[0].number_of_nodes())
        test_mask = index_to_mask(rest_index, size=data[0].number_of_nodes())

        return train_mask, val_mask, test_mask

    if dataname == 'computer':
        dataset = AmazonCoBuyComputerDataset()
    elif dataname == 'photo':
        dataset = AmazonCoBuyPhotoDataset()
    else:
        raise ValueError('Unknown dataset')

    g = dataset[0]
    num_class = dataset.num_classes
    feat = g.ndata['feat']  # get node feature
    label = g.ndata['label']  # get node labels
    train_mask, val_mask, test_mask = random_amazon_splits(dataset, num_class)

    return g, feat, label, train_mask, val_mask, test_mask, num_class
# def main():
#
#
# if __name__ == "__main__":
#     main()
