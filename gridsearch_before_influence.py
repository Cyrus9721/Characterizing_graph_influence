import numpy as np
import pandas as pd
import torch
from dgl import function as fn
from sklearn.preprocessing import OneHotEncoder
from dataset import load_graph_dataset
from tqdm import tqdm
from model_softmax import SimplifiedGraphNeuralNetwork
import os
import argparse
from sklearnex import patch_sklearn

patch_sklearn()


def parse_args():
    parser = argparse.ArgumentParser(description='swtich dataset name')
    parser.add_argument('--dataname', type=str, default='cora',
                        help='dataname to be tunned')
    parser.add_argument('--num_layer', type=int, default=2,
                    help='Number of epochs to train each model for (default: 10)')
    args = parser.parse_args()
    return args


def grid_search(dataname, l2_reg, num_layer):
    # read in the data
    graph, feat, labels, train_mask, val_mask, test_mask, number_classes = load_graph_dataset(dataname)

    # node preprocessing
    feat0 = feat.clone()
    degs = graph.in_degrees().float().clamp(min=1)
    norm = torch.pow(degs, -0.5)
    norm = norm.to(feat0.device).unsqueeze(1)

    for _ in range(num_layer):
        feat0 = feat0 * norm
        graph.ndata['h'] = feat0
        graph.update_all(fn.copy_u('h', 'm'),
                         fn.sum('m', 'h'))
        feat0 = graph.ndata.pop('h')
        feat0 = feat0 * norm

    # generate train set and validation set
    train_x = feat0[train_mask].numpy().astype(np.float64)
    train_y = labels[train_mask].numpy().astype(np.float64)

    val_x = feat0[val_mask].numpy().astype(np.float64)
    val_y = labels[val_mask].numpy().astype(np.float64)

    enc = OneHotEncoder(handle_unknown='ignore')
    enc.fit(train_y.reshape(-1, 1))

    # grid search for parameter tuning
    acc2 = []
    l = []
    for i in tqdm(l2_reg):
        lr = SimplifiedGraphNeuralNetwork(l2_reg=i, fit_intercept=True)
        lr.fit(train_x, train_y, sample_weight=None, verbose=False)
        acc2.append(np.mean(lr.model.predict(val_x) == val_y))
        l.append(i)

    idx = np.where(acc2 == max(acc2))[0][0]
    print('first grid search result:')
    print('validation accuracy:', np.array(acc2)[idx])
    print('weight decay:', np.array(l)[idx])

    return np.array(acc2)[idx], np.array(l)[idx]


def main():
    l2_reg_epoch_1 = [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 5, 10]
    l2_reg_epoch_2 = np.arange(0.001, 0.01, 0.001)
    args = parse_args()
    acc, l2 = grid_search(dataname=args.dataname, l2_reg=l2_reg_epoch_2, num_layer= args.num_layer)
    df = pd.DataFrame([l2]).T
    df.columns = ['l2_reg']
    df.to_csv(os.path.join('hyper_parameter', args.dataname + '.csv'))


if __name__ == "__main__":
    main()
