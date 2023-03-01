import torch
import dgl
from dgl import function as fn
from dgl.base import DGLError
import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", message="DGLGraph\.__len__")


class EdgeInfluenceSGC:

    def __init__(self, graph, feature, from_index, to_index, k=2, parameter=None):
        self.graph = graph
        self.feature = feature
        self.from_index = from_index
        self.to_index = to_index
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
        # i, j = torch.tensor(self.from_index), torch.tensor(self.to_index)
        i = self.from_index
        j = self.to_index
        if (i in from_indexes) and (j in to_indexes):
            graph_0 = dgl.graph((from_indexes, to_indexes))
            # removed_edge_id = graph_0.edge_id(i, j)
            removed_edge_id = graph_0.edge_ids(i, j)
            graph_0.remove_edges(torch.tensor([removed_edge_id]))
            self.removed_graph = graph_0
        else:
            raise DGLError('The input edges does not exists')

    def remove_edges_sgc_from_influence(self):

        graph_0 = self.graph.clone()

        edge_ids1 = graph_0.edge_ids(self.from_index, self.to_index)
        edge_ids2 = graph_0.edge_ids(self.to_index, self.from_index)
        edge_ids = torch.concat([edge_ids1, edge_ids2])
        graph_0.remove_edges(edge_ids)

        self.removed_graph = graph_0

    def add_edges_sgc_from_influence(self):

        graph_0 = self.graph.clone()

        graph_0.add_edges(self.from_index, self.to_index)
        graph_0.add_edges(self.to_index, self.from_index)
        
        self.removed_graph = graph_0


    def remove_edges_sgc_from_influence_single(self):

        graph_0 = self.graph.clone()

        edge_ids1 = graph_0.edge_ids(self.from_index, self.to_index)

        graph_0.remove_edges(edge_ids1)

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


def generate_remove_index_train(from_indexes, to_indexes, train_mask, seed_val=10):
    torch.manual_seed(seed_val)
    train_index = torch.where(train_mask == 1)[0]
    remove_from_list = torch.zeros(len(train_index), dtype=torch.int64)
    remove_to_list = torch.zeros(len(train_index), dtype=torch.int64)
    for i in range(len(train_index)):
        f_index = train_index[i]

        to_index_list = torch.where(from_indexes == f_index)[0]

        random_index = torch.randint(0, len(to_index_list), (1,))[0]

        j = to_index_list[random_index]

        t_index = to_indexes[j]

        remove_from_list[i] = f_index
        remove_to_list[i] = t_index

    return remove_from_list, remove_to_list

