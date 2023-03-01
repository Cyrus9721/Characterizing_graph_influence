# Code for paper 'Characterizing the influence of graph elements'

This code replicates the experiments from the following paper:
> Zizhang Chen, Peizhao Li, Hongfu Liu, Pengyu Hong
>
> [Characterizing the influence of graph elements](https://arxiv.org/abs/2210.07441)

## Abstract

Influence function, a method from robust statistics, measures the changes of model parameters or some functions about model parameters concerning the removal or modification of training instances. It is an efficient and useful post-hoc method for studying the interpretability of machine learning models without the need for expensive model re-training. Recently, graph convolution networks (GCNs), which operate on graph data, have attracted a great deal of attention. However, there is no preceding research on the influence functions of GCNs to shed light on the effects of removing training nodes/edges from an input graph. Since the nodes/edges in a graph are interdependent in GCNs, it is challenging to derive influence functions for GCNs. To fill this gap, we started with the simple graph convolution (SGC) model that operates on an attributed graph and formulated an influence function to approximate the changes in model parameters when a node or an edge is removed from an attributed graph. Moreover, we theoretically analyzed the error bound of the estimated influence of removing an edge. We experimentally validated the accuracy and effectiveness of our influence estimation function. In addition, we showed that the influence function of an SGC model could be used to estimate the impact of removing training nodes/edges on the test performance of the SGC without re-training the model. Finally, we demonstrated how to use influence functions to guide the adversarial attacks on GCNs effectively.

#### Generate the node feature influence: <br />
```
python calculate_feature_influence.py --dataname dataname --l2_reg l2_regularlization_term
```
For example:
```
python calculate_feature_influence.py --dataname cora --l2_reg 0.01
```

#### Generate the edge influence: <br />
```
python calculate_edge_influence.py --dataname dataname --l2_reg l2_regularlization_term
```
For example:
```
python calculate_edge_influence.py --dataname cora --l2_reg 0.01
```

#### Generate the complete training node influence: <br />
```
python calculate_node_influence.py --dataname dataname --l2_reg l2_regularlization_term
```
For example:
```
python calculate_node_influence.py --dataname cora --l2_reg 0.01
```
<br />

![git](/figures/small_dataset_influence_low_resolution.png) <br />

More to come.
