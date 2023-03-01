import sys

sys.path.append('..')
import numpy as np
import pandas as pd
import torch
import os
import dgl
from dgl import function as fn
from dataset import load_graph_dataset
from model_softmax import SimplifiedGraphNeuralNetwork, fast_hess, fast_hess_cuda, fast_get_inv_hvp_cuda
from gcn_with_node_flipping import gcn_with_node_flipping
import tensorflow.compat.v1 as tf
from graph_neural_networks import SGC_layer1, SGC_layer2
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import log_loss
from scipy.special import softmax, log_softmax
from scipy.linalg import cho_solve, cho_factor
from tqdm import tqdm
import cupy as cp
from random import choice
import heapq

dataname = 'cora'
l2_regularlization_term = 0.01
num_layer = 2

# dataname = 'pubmed'
# l2_regularlization_term = 0.004
# num_layer = 2

# dataname = 'citeseer'
# l2_regularlization_term = 0.003
# num_layer = 2

"""set up random seed, perturb ratio"""
perturb_ratio_list = [0.05, 0.1, 0.15, 0.2]
some_seed_list = [1, 11, 15, 42, 100]
num_times_running = 5

acctual_influence_1 = []
acctual_influence_2 = []

predict_influence_1 = []
predict_influence_2 = []


def index_to_mask(index, size):
    mask = torch.zeros(size, dtype=torch.bool, device=index.device)
    mask[index] = 1
    return mask


def random_splits_label_flip_attack(graph, labels, num_classes, seed):
    # Set new random planetoid splits:
    # * 20 * num_classes labels for training
    # * 500 labels for validation
    # * 1000 labels for testing
    torch.manual_seed(seed)

    indices = []

    for i in range(num_classes):
        index = (labels == i).nonzero().view(-1)
        index = index[torch.randperm(index.size(0))]
        indices.append(index)

    train_index = torch.cat([i[:20] for i in indices], dim=0)

    rest_index = torch.cat([i[20:] for i in indices], dim=0)
    rest_index = rest_index[torch.randperm(rest_index.size(0))]

    train_mask = index_to_mask(train_index, size=graph.num_nodes())
    val_mask = index_to_mask(rest_index[:500], size=graph.num_nodes())
    test_mask = index_to_mask(rest_index[500:1500], size=graph.num_nodes())

    return train_mask, val_mask, test_mask


def get_first_two_frequent(labels):
    class_counts = np.bincount(labels)
    a = np.argsort(class_counts)[-1]
    b = np.argsort(class_counts)[-2]
    return a, b


def node_flipping_attack_rev(dataname=dataname, l2_regularlization_term=0.01, perturb_ratio=0.05,
                             num_layer=2, some_seed=42):
    graph, feat, labels, _, _, _, number_classes = load_graph_dataset(dataname)
    train_mask, val_mask, test_mask = random_splits_label_flip_attack(graph, labels, number_classes, seed=some_seed)

    lr = SimplifiedGraphNeuralNetwork(l2_reg=l2_regularlization_term, fit_intercept=True)

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

    train_x = feat0[train_mask].numpy().astype(np.float32)
    train_y = labels[train_mask].numpy().astype(np.float32)

    test_x = feat0[test_mask].numpy().astype(np.float32)
    test_y = labels[test_mask].numpy().astype(np.float32)

    enc = OneHotEncoder(handle_unknown='ignore')
    enc.fit(train_y.reshape(-1, 1))

    one_hot_labels_train = enc.transform(train_y.reshape(-1, 1)).toarray()
    one_hot_labels_test = enc.transform(test_y.reshape(-1, 1)).toarray()

    """ Train Logistic Regression """
    lr = SimplifiedGraphNeuralNetwork(l2_reg=l2_regularlization_term, fit_intercept=True)
    lr.fit(train_x, train_y, sample_weight=None, verbose=False)
    logits_test_y = test_x @ lr.model.coef_.T + lr.model.intercept_
    logits_train_y = train_x @ lr.model.coef_.T + lr.model.intercept_

    ori_val_loss, ave_ori_val_loss = lr.log_loss(logits_test_y, one_hot_labels_test, l2_reg=True)

    numpy_theoritic_loss = log_loss(test_y, softmax(logits_test_y, axis=1))

    train_total_grad, train_indiv_grad = lr.grad(train_x, logits_train_y,
                                                 one_hot_labels_train, l2_reg=True)
    val_loss_total_grad, val_loss_indiv_grad = lr.grad(test_x, logits_test_y,
                                                       one_hot_labels_test, l2_reg=True)

    # hessian_no_reg, hess, hessian_reg_term = lr.hess(train_x, logits_train_y)
    # hess = fast_hess_cuda(train_x, logits_train_y)
    hess = lr.hess_cuda(train_x, logits_train_y)

    loss_grad_hvp = fast_get_inv_hvp_cuda(hess, val_loss_total_grad.T, cholskey=True)
    # loss_grad_hvp = fast_get_inv_hvp_cuda(hess, val_loss_total_grad.T, cholskey=False)
    loss_grad_hvp = cp.asnumpy(loss_grad_hvp)

    pred_infl = train_indiv_grad.dot(loss_grad_hvp)

    pred_infl = list(pred_infl.reshape(-1))
    #
    num_train = len(train_x)
    act_infl = []

    for i in tqdm(range(num_train)):
        lr_new = SimplifiedGraphNeuralNetwork(l2_reg=l2_regularlization_term, fit_intercept=True)
        train_x_new = np.delete(train_x, i, axis=0)
        train_y_new = np.delete(train_y, i)
        lr_new.fit(train_x_new, train_y_new)

        logits_test_y_new = test_x @ lr_new.model.coef_.T + lr_new.model.intercept_

        new_ori_val_loss, new_ave_ori_val_loss = lr_new.log_loss(logits_test_y_new, one_hot_labels_test, l2_reg=True)
        act_infl.append(new_ori_val_loss - ori_val_loss)

    df = pd.DataFrame([pred_infl, act_infl]).T
    df.columns = ['predicted influence', 'actual influence']
    predicted_influence = np.array(pred_infl)

    a, b = get_first_two_frequent(labels[test_mask])
    idx1 = np.where(labels[train_mask].numpy() == a)[0]
    idx2 = np.where(labels[train_mask].numpy() == b)[0]

    num_train = np.sum(train_mask.numpy() == 1)
    num_perturb = int(perturb_ratio * len(idx2) * 2)
    predicted_influence_combined = np.concatenate([predicted_influence[idx1], predicted_influence[idx2]])
    predicted_influence_combined_sorted = np.sort(predicted_influence_combined)[::-1]
    threshold = predicted_influence_combined_sorted[num_perturb]

    perturbed_train_y = train_y.copy()
    new_labels_a = np.repeat(a, len(idx1))
    new_labels_b = np.repeat(b, len(idx2))
    assert (len(idx1) == len(idx2))

    predicted_influence_combined_sorted

    idx_a_to_b = np.where(predicted_influence[idx1] > threshold)[0]
    idx_b_to_a = np.where(predicted_influence[idx2] > threshold)[0]

    new_labels_a[idx_a_to_b] = b
    new_labels_b[idx_b_to_a] = a

    perturbed_train_y[idx1] = new_labels_a
    perturbed_train_y[idx2] = new_labels_b

    new_labels = labels.numpy().copy()
    new_labels[train_mask] = perturbed_train_y
    new_labels = torch.tensor(new_labels)

    gcn_with_node_flip = gcn_with_node_flipping(graph=graph, features=feat, new_labels=new_labels,
                                                train_mask=train_mask, val_mask=val_mask, test_mask=test_mask,
                                                num_classes=number_classes, dropout=0.5)

    gcn_without_node_flip = gcn_with_node_flipping(graph=graph, features=feat, new_labels=labels,
                                                   train_mask=train_mask, val_mask=val_mask, test_mask=test_mask,
                                                   num_classes=number_classes)

    acc_flip = gcn_with_node_flip.train_evaluate()
    acc_no_flip = gcn_without_node_flip.train_evaluate()

    return acc_flip, acc_no_flip

dataname = 'cora'
l2_regularlization_term = 0.01
num_layer = 2

# dataname = 'pubmed'
# l2_regularlization_term = 0.004
# num_layer = 2

# dataname = 'citeseer'
# l2_regularlization_term = 0.003
# num_layer = 2


perturb_ratio_list = [0.1, 0.15, 0.2]
# some_seed_list = [1, 11, 15, 42, 100]
some_seed_list = [15]
num_times_running = 1

for perturb_ratio in perturb_ratio_list:
    temp_acc_flip_list = []
    temp_acc_no_flip_list = []
    for s in some_seed_list:

        for _ in range(num_times_running):
            acc_flip, acc_no_flip = node_flipping_attack_rev(dataname = dataname, l2_regularlization_term = l2_regularlization_term,
                                                             perturb_ratio = perturb_ratio, num_layer = num_layer,
                                                             some_seed = s)

            temp_acc_flip_list.append(acc_flip)
            temp_acc_no_flip_list.append(acc_no_flip)

    flip_df = pd.DataFrame([temp_acc_flip_list, temp_acc_no_flip_list]).T
    flip_df.columns = ['filped accuracy', 'original accuracy']
    flip_df.to_csv('result_flip_attack/'+ dataname + '/'+ 'new_perturb_ratio_' + str(perturb_ratio)+'.csv', index = False)
