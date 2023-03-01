import numpy as np
import pandas as pd
import copy
import torch
import dgl
import cupy as cp
import collections
import random
from dgl import function as fn
from sklearn.preprocessing import OneHotEncoder
from dataset import load_graph_dataset
from tqdm import tqdm
from model_softmax import SimplifiedGraphNeuralNetwork
from sklearn.linear_model import LogisticRegression

from torch_geometric.datasets import Planetoid
from pygod.utils import gen_attribute_outliers
from pygod.models import CoLA, DOMINANT
from dgl.data import FraudDataset
from pygod.utils.metric import eval_roc_auc, eval_recall_at_k, eval_precision_at_k
from sklearn.metrics import precision_score, accuracy_score
from model_softmax import SimplifiedGraphNeuralNetwork, fast_hess, fast_hess_cuda, fast_get_inv_hvp_cuda
from tqdm import tqdm
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.neighbors import NearestCentroid

import itertools


# generate node feature outliers by classes
# generate outliers for 'class1' from 'class2'
# generate outliers for 'class2 from 'class1'
def generate_node_feature_outliers(features, labels, class1, class2, num_class_1_to_2, num_class_2_to_1):
    class_class1 = class1
    class_class2 = class2

    k_class1 = num_class_1_to_2  #
    k_class2 = num_class_2_to_1

    # index of corresponding class
    class_idx_class1 = torch.where(labels == class_class1)[0]
    class_idx_class2 = torch.where(labels == class_class2)[0]

    # feature of corresponding class
    feat_class1 = features[class_idx_class1]
    feat_class2 = features[class_idx_class2]

    # label of corresponding class
    label_class1 = labels[class_idx_class1]
    label_class2 = labels[class_idx_class2]

    # find the centroid of each classes
    clf = NearestCentroid()
    clf.fit(np.concatenate([feat_class1, feat_class2]), np.concatenate([label_class1, label_class2]))

    centroid_class1, centroid_class2 = clf.centroids_

    # class2 the euclidean distance between centriod and features for each class
    dist_class1 = euclidean_distances(centroid_class1.reshape(1, -1), np.array(feat_class1)).reshape(-1)
    dist_class2 = euclidean_distances(centroid_class2.reshape(1, -1), np.array(feat_class2)).reshape(-1)

    # class2 k node features by sorting the cuclidean distance
    # features from the class1 class to be converted to class2
    idx_class1_k = np.argsort(dist_class1)[0:k_class1]
    feat_class1_to_class2 = feat_class1[idx_class1_k]

    idx_class2_k = np.argsort(dist_class2)[0:k_class2]
    feat_class2_to_class1 = feat_class2[idx_class2_k]

    # generated the converted labels and the original labels
    # class1
    label_class1_k_orig = label_class1[0:k_class1]  # original label
    label_class1_k_convert = label_class2[0:k_class1]  # converted label
    # class2
    label_class2_k_orig = label_class2[0:k_class2]
    label_class2_k_convert = label_class2[0:k_class2]

    return (feat_class1_to_class2, label_class1_k_orig,
            label_class1_k_convert), (feat_class2_to_class1, label_class2_k_orig, label_class2_k_convert)


# return list of list of nodes that to be connected
def generate_nodes_connected(graph, labels, class_outlier, num_batches=2, random_seed=None):
    # return batch of edges for batch of outliers of certain class
    if random_seed:
        random.seed(random_seed)

    source_index, target_index = graph.edges()

    # label that the feature changed to
    label_changed = class_outlier

    # index of labels changed
    node_idx_changed = torch.where(labels == label_changed)[0]

    # get the index of edges that the source, target node is the changed class
    edge_idx_source_node = np.where(np.in1d(source_index, node_idx_changed))[0]
    edge_idx_target_node = np.where(np.in1d(target_index, node_idx_changed))[0]
    edge_idx_all_node = np.unique(np.concatenate([edge_idx_source_node, edge_idx_target_node]))

    source_node_list = np.array(source_index[edge_idx_all_node])
    target_node_list = np.array(target_index[edge_idx_all_node])

    source_node_list_no_self_loop = source_node_list[source_node_list != target_node_list]
    target_node_list_no_self_loop = target_node_list[source_node_list != target_node_list]

    # arr nodes that are connect to n
    arr = source_node_list_no_self_loop[target_node_list_no_self_loop == np.array(node_idx_changed[0])]
    for i in range(1, len(node_idx_changed)):
        arr = np.append(arr,
                        source_node_list_no_self_loop[target_node_list_no_self_loop == np.array(node_idx_changed[i])])

    collection_nodes = collections.Counter(arr)

    nodes_potential = np.array(list(collection_nodes.keys()))
    nodes_frequency = np.array(list(collection_nodes.values()))
    p_nodes = nodes_frequency / np.sum(nodes_frequency)

    average_degree = int(len(arr) / len(node_idx_changed))

    batches_node_list = []
    for j in range(num_batches):
        num_chose = random.choice(nodes_frequency)
        node_list = np.random.choice(nodes_potential, num_chose, p=p_nodes)
        batches_node_list.append(list(node_list))
    return batches_node_list


def get_only_in_node_id(graph, labels, class1):
    source_index, target_index = graph.edges()

    node_idx = np.array(torch.where(labels == class1)[0])

    edge_idx_source_node = np.where(np.in1d(source_index, node_idx))[0]

    edge_idx_target_node = np.where(np.in1d(target_index, node_idx))[0]

    edge_idx_all_node = np.unique(np.concatenate([edge_idx_source_node, edge_idx_target_node]))

    source_node_list = np.array(source_index[edge_idx_all_node])
    target_node_list = np.array(target_index[edge_idx_all_node])

    source_node_list_no_self_loop = source_node_list[source_node_list != target_node_list]
    target_node_list_no_self_loop = target_node_list[source_node_list != target_node_list]

    node_only_within_class = []
    for i in range(len(source_node_list_no_self_loop)):
        if (source_node_list_no_self_loop[i] in node_idx) and (target_node_list_no_self_loop[i] in node_idx):
            node_only_within_class.append(source_node_list_no_self_loop[i])
    node_only_within_class = np.unique(node_only_within_class)
    return node_only_within_class


def generate_edges_outliers_numbers_list(a, k):
    arr = np.repeat(np.ceil(a / k).astype(int), k)
    arr[-1] += (a - np.ceil(a / k).astype(int) * k)
    return arr


def generate_outlier_edge_random(labels, labels_orig, labels_outlier, num_edges, random_seed=None):
    if random_seed:
        random.seed(random_seed)
    node_idx_orig = torch.where(labels == labels_orig)[0]
    node_idx_outlier = torch.where(labels == labels_outlier)[0]

    from_idx = random.choices(node_idx_orig, k=num_edges)
    to_idx = random.choices(node_idx_outlier, k=num_edges)

    return from_idx, to_idx


def generate_edge_outlier_from_within_class(graph, labels, labels_orig, labels_outlier, num_edges, random_seed=None):
    if random_seed:
        random.seed(random_seed)

    node_only_within_class_orig = get_only_in_node_id(graph, labels, labels_orig)
    node_only_within_class_outlier = get_only_in_node_id(graph, labels, labels_outlier)

    from_idx = random.choices(node_only_within_class_orig, k=num_edges)
    to_idx = random.choices(node_only_within_class_outlier, k=num_edges)

    return from_idx, to_idx


def inject_edge_outlier(graph, labels, num_edges=42, random_seed=42):
    graph0 = graph.clone()
    labels_unique = np.unique(labels)
    comb_labels = list(itertools.combinations(labels_unique, 2))
    comb_numbers = generate_edges_outliers_numbers_list(num_edges, len(comb_labels))
    labels_edges_outliers = np.zeros(len(graph0.edges()[0])).astype(int)

    for i in range(len(comb_labels)):
        num_edges_temp = comb_numbers[i]
        class1, class2 = comb_labels[i]
        source_node, target_node = generate_edge_outlier_from_within_class(graph, labels, class1, class2,
                                                                           num_edges_temp, random_seed=random_seed)
        graph0.add_edges(source_node, target_node)
        graph0.add_edges(target_node, source_node)

        for j in range(len(source_node)):
            labels_edges_outliers = np.append(labels_edges_outliers, 1)
            labels_edges_outliers = np.append(labels_edges_outliers, 1)

    return graph0, labels_edges_outliers
