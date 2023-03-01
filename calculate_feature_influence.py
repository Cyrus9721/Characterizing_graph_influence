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
    parser.add_argument('--out_dir', type =str, default='',
                        help='output directory')
    args = parser.parse_args()
    return args


def generate_node_feature_influence(dataname, l2_regularlization_term, num_layer=2):
    graph, feat, labels, train_mask, val_mask, test_mask, number_classes = load_graph_dataset(dataname)

    """feature propagation"""
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

    """generate training and validation set"""
    train_x = feat0[train_mask].numpy().astype(np.float32)
    train_y = labels[train_mask].numpy().astype(np.float32)

    val_x = feat0[val_mask].numpy().astype(np.float32)
    val_y = labels[val_mask].numpy().astype(np.float32)

    enc = OneHotEncoder(handle_unknown='ignore')
    enc.fit(train_y.reshape(-1, 1))

    one_hot_labels_train = enc.transform(train_y.reshape(-1, 1)).toarray()
    one_hot_labels_val = enc.transform(val_y.reshape(-1, 1)).toarray()

    """ Train Simplifying Graph Neural Network"""
    lr = SimplifiedGraphNeuralNetwork(l2_reg=l2_regularlization_term, fit_intercept=True)
    lr.fit(train_x, train_y, sample_weight=None, verbose=False)
    logits_val_y = val_x @ lr.model.coef_.T + lr.model.intercept_
    logits_train_y = train_x @ lr.model.coef_.T + lr.model.intercept_

    ori_val_loss, ave_ori_val_loss = lr.log_loss(logits_val_y, one_hot_labels_val, l2_reg=True)

    train_total_grad, train_indiv_grad = lr.grad(train_x, logits_train_y,
                                                 one_hot_labels_train, l2_reg=True)
    val_loss_total_grad, val_loss_indiv_grad = lr.grad(val_x, logits_val_y,
                                                       one_hot_labels_val, l2_reg=True)

    hess = lr.hess_cuda(train_x, logits_train_y)

    loss_grad_hvp = fast_get_inv_hvp_cuda(hess, val_loss_total_grad.T, cholskey=True)

    loss_grad_hvp = cp.asnumpy(loss_grad_hvp)

    pred_infl = train_indiv_grad.dot(loss_grad_hvp)

    pred_infl = list(pred_infl.reshape(-1))

    num_train = len(train_x)

    act_infl = []

    """retrain the model to verify the loss change"""
    print('Retraining the model to verify loss change')
    for i in tqdm(range(num_train)):
        lr_new = SimplifiedGraphNeuralNetwork(l2_reg=l2_regularlization_term, fit_intercept=True)
        train_x_new = np.delete(train_x, i, axis=0)
        train_y_new = np.delete(train_y, i)
        lr_new.fit(train_x_new, train_y_new)

        logits_val_y_new = val_x @ lr_new.model.coef_.T + lr_new.model.intercept_

        new_ori_val_loss, new_ave_ori_val_loss = lr_new.log_loss(logits_val_y_new, one_hot_labels_val, l2_reg=True)
        act_infl.append(new_ori_val_loss - ori_val_loss)

    return pred_infl, act_infl


def plot_infl(pred_infl, act_infl, dataname, dir='task1'):
    low_limit = min(np.min(pred_infl), np.min(act_infl)) - 2
    up_limit = max(np.max(pred_infl), np.max(act_infl)) + 2
    x = np.linspace(low_limit, up_limit)
    plt.plot(x, x)
    plt.plot(act_infl, pred_infl, 'o', color='blue')
    plt.xlabel('actual change in loss')
    titlename = 'Influence function on ' + dataname + ' dataset - Node Feature'
    plt.ylabel('predicted change in loss')
    plt.title(titlename)
    plt.savefig(os.path.join(dir, titlename + '.png'))
    plt.show()


def main(use_default=True):
    args = parse_args()
    if use_default:
        l2_regularlization_term = args.l2_reg
    else:
        df = pd.read_csv(os.path.join('hyper_parameter', args.dataname + '.csv'))
        l2_regularlization_term = df.l2_reg.values[0]

    pred_infl, act_infl = generate_node_feature_influence(args.dataname, l2_regularlization_term, num_layer=2)
    plot_infl(pred_infl, act_infl, args.dataname, dir='task1')
    df_infl = pd.DataFrame([act_infl, pred_infl]).T
    df_infl.columns = ['actual influence', 'predict influence']
    df_infl.to_csv('influence_score/' + args.dataname + '_node_feature_influence.csv', index=False)


if __name__ == "__main__":
    main()
