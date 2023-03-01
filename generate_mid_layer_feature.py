import torch
import dgl
import numpy as np
import pandas as pd
import torch.nn as nn
from graph_neural_networks import SGC_layer2, SGC_layer1
from dataset import load_graph_dataset

"""
For two layer simplified graph neural network. Generate features for the middle layer. 
First fit a two-lay simplified graph neural network, generate the output of the first layer, 
"""


def evaluate_accuracy(model, g, features, labels, mask):
    model.eval()
    with torch.no_grad():
        logits = model(g, features)
        logits = logits[mask]
        labels = labels[mask]
        _, indices = torch.max(logits, dim=1)
        correct = torch.sum(indices == labels)
        return correct.item() * 1.0 / len(labels)


class FeatureExtraction:

    def __init__(self, num_layers=2, num_iter=100, lr=0.2, weight_decay=5e-6, hidden_feat=20,
                 bias=False, device="cuda:0", dataset='cora', seed=1):

        self.num_layers = num_layers
        self.num_iter = num_iter
        self.lr = lr
        self.weight_decay = weight_decay
        self.hidden_feat = hidden_feat
        self.bias = bias
        self.device = device
        self.dataset_name = dataset

        self.model = None
        self.train_idx = None
        self.val_idx = None
        self.test_idx = None
        self.labels = None
        self.seed = seed

    def extract_feature(self, index_remove=None):
        torch.manual_seed(self.seed)

        graph, feat, labels, train_mask, val_mask, test_mask, num_labels = load_graph_dataset(self.dataset_name)

        model = SGC_layer2(in_feats=feat.shape[1],
                           hidden_feats=self.hidden_feat,
                           out_feats=num_labels,
                           k=self.num_layers,
                           bias=self.bias,
                           index_remove=None)

        # model = SGC_layer1(in_feats=feat.shape[1],
        #                    out_feats=num_labels,
        #                    k=self.num_layers)

        if self.device != 'cpu':
            model.cuda()

        loss_fcn = torch.nn.CrossEntropyLoss()

        optimizer = torch.optim.Adam(model.parameters(),
                                     lr=self.lr,
                                     weight_decay=self.weight_decay)

        for epoch in range(self.num_iter):
            model.train()
            logits = model(graph, feat)
            loss = loss_fcn(logits[train_mask], labels[train_mask])
            loss.backward()
            optimizer.step()

        acc = evaluate_accuracy(model, graph, feat, labels, test_mask)
        print('accuracy:', acc)
        self.model = model
        self.train_idx = train_mask.numpy()
        self.val_idx = val_mask.numpy()
        self.test_idx = test_mask.numpy()
        self.labels = labels.numpy()

    def preprocessed_data(self, save=None):

        _, embed2 = self.model.get_embed()
        # embed2 = self.model.get_embed()

        train_x = embed2[self.train_idx]
        val_x = embed2[self.val_idx]
        test_x = embed2[self.test_idx]

        train_y = self.labels[self.train_idx]
        val_y = self.labels[self.val_idx]
        test_y = self.labels[self.test_idx]

        if save is not None:
            df = np.concatenate([self.labels.reshape(-1, 1),
                                 embed2,
                                 self.train_idx.reshape(-1, 1),
                                 self.val_idx.reshape(-1, 1),
                                 self.test_idx.reshape(-1, 1)], axis=1)
            pd.DataFrame(df).to_csv(save, columns=None, index=False, header=None)
        return train_x, train_y, val_x, val_y, test_x, test_y


def main():
    FeatureExtractor = FeatureExtraction(num_layers=2, num_iter=100, lr=0.02, hidden_feat=20, device='cpu',
                                         dataset='cora')
    FeatureExtractor.extract_feature()
    FeatureExtractor.preprocessed_data(save='feat.csv')


if __name__ == '__main__':
    main()
