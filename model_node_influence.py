import torch
import dgl
from dgl import function as fn
from dgl.base import DGLError
import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", message="DGLGraph\.__len__")


class NodeInfluenceSGC:

    def __init__(self, graph, feature, node_index, k=2, parameter=None):
        self.graph = graph
        self.feature = feature
        self.node_index = node_index
        self.k = k

        self.removed_graph = None
        self.feature_new = None
        self.feature_different = None
        self.parameter = parameter

    def list_edges_sgc(self):
        from_indexes, to_indexes = self.graph.edges()
        return from_indexes, to_indexes

    def remove_edges_sgc(self):
        from_indexes, to_indexes = self.list_edges_sgc()
        node_id = self.node_index
        node_from_indexes = torch.where(from_indexes == node_id)[0]
        node_to_indexes = torch.where(to_indexes == node_id)[0]
        node_all_indexes = torch.unique(torch.concat([node_from_indexes, node_to_indexes]))
        graph_0 = self.graph.clone()
        graph_0.remove_edges(node_all_indexes)
        self.removed_graph = graph_0

    def calculate_modified_features(self):

        graph_removed = self.removed_graph
        feature_remove = self.feature.clone()

        degree_remove = graph_removed.in_degrees().float().clamp(min=1)
        norm_removed = torch.pow(degree_remove, -0.5)
        norm_removed = norm_removed.to(feature_remove.device).unsqueeze(1)

        for _ in range(self.k):
            feature_remove = feature_remove * norm_removed
            graph_removed.ndata['h'] = feature_remove
            graph_removed.update_all(fn.copy_u('h', 'm'), fn.sum('m', 'h'))
            feature_remove = graph_removed.ndata.pop('h')
            feature_remove = feature_remove * norm_removed

        self.feature_new = feature_remove
        return feature_remove
