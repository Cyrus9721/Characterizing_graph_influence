import sys

sys.path.append('..')
import numpy as np
import pandas as pd
import os
import torch
import cupy as cp
import argparse
from dgl import function as fn
from sklearn.preprocessing import OneHotEncoder
from dataset import load_graph_dataset
from model_node_influence import NodeInfluenceSGC
from model_softmax import SimplifiedGraphNeuralNetwork, fast_hess, fast_hess_cuda, fast_get_inv_hvp_cuda
from matplotlib import pyplot as plt
from tqdm import tqdm
from sklearnex import patch_sklearn

patch_sklearn()


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


"""retrieve all edges connected to traing nodes"""


def generate_remove_index_train_all(from_indexes, to_indexes, train_mask):
    train_index = torch.where(train_mask == 1)[0]
    remove_from_list = []
    remove_to_list = []
    for i in tqdm(range(len(train_index))):
        f_index = train_index[i]
        to_index_list = torch.where(from_indexes == f_index)[0]
        for to_index_e in to_index_list:
            j = to_index_e
            t_index = to_indexes[j]

            remove_from_list.append(f_index)
            remove_to_list.append(t_index)

    return torch.tensor(remove_from_list), torch.tensor(remove_to_list)


def plot_infl(pred_infl, act_infl, dataname, dir='task1'):
    low_limit = min(np.min(pred_infl), np.min(act_infl)) - 2
    up_limit = max(np.max(pred_infl), np.max(act_infl)) + 2
    x = np.linspace(low_limit, up_limit)
    plt.plot(x, x)
    plt.plot(pred_infl, act_infl, 'o', color='blue')
    plt.xlabel('actual change in loss')
    titlename = 'Influence function on ' + dataname + ' dataset - Complete Node'
    plt.ylabel('predicted change in loss')
    plt.title(titlename)
    plt.show()


def generate_complete_node_influence(graph, feat, labels, train_mask, val_mask, test_mask, number_classes,
                                     l2_regularlization_term, num_layer=2):
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

    val_x = feat0[val_mask].numpy().astype(np.float32)
    val_y = labels[val_mask].numpy().astype(np.float32)

    train_node_idx = torch.where(train_mask == 1)[0]

    # convert to one-hot labels
    enc = OneHotEncoder(handle_unknown='ignore')
    enc.fit(train_y.reshape(-1, 1))

    one_hot_labels_train = enc.transform(train_y.reshape(-1, 1)).toarray()
    one_hot_labels_val = enc.transform(val_y.reshape(-1, 1)).toarray()

    from_indexes, to_indexes = graph.edges()

    f_l, t_l = generate_remove_index_train_all(from_indexes, to_indexes, train_mask)

    acctual_influence_node_features = []
    acctual_influence_edges = []

    predict_influence_node_features = []
    predict_influence_edges = []

    ##### 2.1 train the original data, calculate the hessian matrix
    lr_origin = SimplifiedGraphNeuralNetwork(l2_reg=l2_regularlization_term, fit_intercept=True)

    lr_origin.fit(train_x, train_y, sample_weight=None, verbose=False)

    logits_val_y_origin = val_x @ lr_origin.model.coef_.T + lr_origin.model.intercept_

    logits_train_y_origin = train_x @ lr_origin.model.coef_.T + lr_origin.model.intercept_

    ori_val_loss, ave_ori_val_loss = lr_origin.log_loss(logits_val_y_origin, one_hot_labels_val, l2_reg=True)

    # numpy_theoritic_loss = log_loss(val_y, softmax(logits_val_y_origin, axis=1))
    # # set l2_reg to False, verify the correctness of calculations
    # assert np.allclose(numpy_theoritic_loss, ave_ori_val_loss)

    val_loss_total_grad_orig, val_loss_indiv_grad_orig = lr_origin.grad(val_x,
                                                                        logits_val_y_origin,
                                                                        one_hot_labels_val, l2_reg=True)

    hess = lr_origin.hess_cuda(train_x, logits_train_y_origin, l2_reg=True)

    loss_grad_hvp = fast_get_inv_hvp_cuda(hess, val_loss_total_grad_orig.T, cholskey=True)

    loss_grad_hvp = cp.asnumpy(loss_grad_hvp)
    del hess

    acctual_influence_1 = []
    acctual_influence_2 = []

    predict_influence_1 = []
    predict_influence_2 = []

    for k in tqdm(range(len(train_node_idx))):
        node_id = train_node_idx.numpy()[k]
        nis = NodeInfluenceSGC(graph=graph, feature=feat, node_index=node_id)

        # 2, remove the edges, calculate the perturbated feature
        nis.remove_edges_sgc()
        feat_removed1 = nis.calculate_modified_features()

        extra_index = torch.unique(torch.where(feat0 != feat_removed1)[0])

        extra_index_train = torch.tensor(
            [extra_index[i] for i in range(len(extra_index)) if extra_index[i] in train_node_idx]).numpy()

        extra_index_train_in_train = [
            np.where(train_node_idx.numpy() == extra_index_train[j])[0][0] for j in range(len(extra_index_train))]

        # 1, we need to remove the changed node feature from the perturbated feature,
        # let it not added to the original feature

        """modified node features"""
        extra_index_train_remove_node = extra_index_train.copy()
        relative_node_id = np.where(extra_index_train_remove_node == node_id)[0]
        extra_index_train_remove_node = np.delete(extra_index_train_remove_node, relative_node_id)
        feat_to_be_added = feat_removed1[extra_index_train_remove_node].numpy()

        """index corresponding to modified node features"""
        perturb_index = extra_index_train_in_train
        added_index = perturb_index.copy()
        added_index.remove(k)

        train_x_new = feat_to_be_added
        train_y_new = train_y[added_index]

        train_x_orig = np.concatenate([train_x, train_x_new])
        train_y_orig = np.concatenate([train_y, train_y_new])

        one_hot_labels_train_0 = enc.transform(train_y_orig.reshape(-1, 1)).toarray()
        logits_train_y_origin_0 = train_x_orig @ lr_origin.model.coef_.T + lr_origin.model.intercept_

        train_total_grad_orig, train_indiv_grad_orig = lr_origin.grad(train_x_orig,
                                                                      logits_train_y_origin_0,
                                                                      one_hot_labels_train_0, l2_reg=True)

        # start_time = time.time()
        pred_infl = train_indiv_grad_orig.dot(loss_grad_hvp)
        # time_infl.append(time.time() - start_time)

        weight_3 = np.ones(len(train_x_orig))
        weight_3[perturb_index] = 0  # 1...0...11

        lr_new_2 = SimplifiedGraphNeuralNetwork(l2_reg=l2_regularlization_term, fit_intercept=True)
        train_x_delete_2 = train_x_orig[weight_3 == 1]
        train_y_delete_2 = train_y_orig[weight_3 == 1]

        # start_time = time.time()
        lr_new_2.fit(train_x_delete_2, train_y_delete_2)
        # time_retrain.append(time.time() - start_time)

        logits_val_y_new_2 = val_x @ lr_new_2.model.coef_.T + lr_new_2.model.intercept_
        new_ori_val_loss_2, _ = lr_new_2.log_loss(logits_val_y_new_2, one_hot_labels_val, l2_reg=True)

        predict_influence_1.append(np.sum(pred_infl[perturb_index]) - np.sum(pred_infl[len(train_x):]))
        acctual_influence_1.append(new_ori_val_loss_2 - ori_val_loss)

    return predict_influence_1, acctual_influence_1


def main(use_default=True):
    args = parse_args()
    if use_default:
        l2_regularlization_term = args.l2_reg
    else:
        df = pd.read_csv(os.path.join('hyper_parameter', args.dataname + '.csv'))
        l2_regularlization_term = df.l2_reg.values[0]

    graph, feat, labels, train_mask, val_mask, test_mask, number_classes = load_graph_dataset(args.dataname)

    pred_infl, act_infl = generate_complete_node_influence(graph, feat, labels, train_mask, val_mask, test_mask,
                                                           number_classes, l2_regularlization_term, num_layer=2)
    plot_infl(pred_infl, act_infl, args.dataname, dir='task1')
    df_infl = pd.DataFrame([act_infl, pred_infl]).T
    df_infl.columns = ['actual influence', 'predict influence']
    df_infl.to_csv('influence_score/' + args.dataname + '_node_influence.csv', index=False)


if __name__ == "__main__":
    main()
