import torch
import torch.nn as nn
from dgl import function as fn
from dgl.base import DGLError


class SGC_layer1(nn.Module):
    """
    Original 2-layer Simplified Graph Neural Network
    """

    def __init__(self,
                 in_feats,
                 out_feats,
                 k=2,
                 cached=False,
                 norm=None,
                 allow_zero_in_degree=False):
        super(SGC_layer1, self).__init__()
        self.embed = None
        self.in_feats = in_feats
        self.out_feats = out_feats
        self.fc = nn.Linear(in_feats, out_feats)

        self._cached = cached
        self._cached_h = None
        self._k = k
        self.norm = norm
        self._allow_zero_in_degree = allow_zero_in_degree
        self.reset_parameters()

    def reset_parameters(self):

        nn.init.xavier_uniform_(self.fc.weight)

        if self.fc.bias is not None:
            nn.init.zeros_(self.fc.bias)

    def set_allow_zero_in_degree(self, set_value):

        self._allow_zero_in_degree = set_value

    def forward(self, graph, feat):

        with graph.local_scope():
            if not self._allow_zero_in_degree:
                if (graph.in_degrees() == 0).any():
                    raise DGLError('There are 0-in-degree nodes in the graph, '
                                   'output for those nodes will be invalid. '
                                   'This is harmful for some applications, '
                                   'causing silent performance regression. '
                                   'Adding self-loop on the input graph by '
                                   'calling `g = dgl.add_self_loop(g)` will resolve '
                                   'the issue. Setting ``allow_zero_in_degree`` '
                                   'to be `True` when constructing this module will '
                                   'suppress the check and let the code run.')

            if self._cached_h is not None:
                feat = self._cached_h
            else:
                # compute normalization
                degs = graph.in_degrees().float().clamp(min=1)
                norm = torch.pow(degs, -0.5)
                norm = norm.to(feat.device).unsqueeze(1)
                # compute (D^-1 A^k D)^k X
                for _ in range(self._k):
                    feat = feat * norm
                    graph.ndata['h'] = feat
                    graph.update_all(fn.copy_u('h', 'm'),
                                     fn.sum('m', 'h'))
                    feat = graph.ndata.pop('h')
                    feat = feat * norm

                # normalize the feature
                if self.norm is not None:
                    feat = self.norm(feat)

                # cache feature
                if self._cached:
                    self._cached_h = feat

                out = self.fc(feat)

                self.embed = feat
                return out

    def num_parameters(self):
        return self.fc.weight.nelement()

    def get_weight(self):
        return self.fc.weight.data.numpy()

    def get_embed(self):
        return self.embed.numpy()


class SGC_layer2(nn.Module):
    """
    2-layer Simplified Graph Neural Network
    The feature from the layer is extracted,
    Theta = Theta1 @ Theta2
    """

    def __init__(self,
                 in_feats,
                 hidden_feats,
                 out_feats,
                 k=2,
                 cached=False,
                 bias=True,
                 norm=None,
                 allow_zero_in_degree=False,
                 index_remove=None):
        super(SGC_layer2, self).__init__()
        # input, hidden, output feature
        self.embed1 = None
        self.embed2 = None
        self.in_feats = in_feats
        self.hidden_feats = hidden_feats
        self.out_feats = out_feats

        # network architecture
        self.fc1 = nn.Linear(in_feats, hidden_feats, bias=bias)
        self.fc2 = nn.Linear(hidden_feats, out_feats, bias=bias)

        self._cached = cached
        self._cached_h = None
        self._k = k
        self.norm = norm
        self._allow_zero_in_degree = allow_zero_in_degree
        self.reset_parameters()
        self.index_remove = index_remove
        self.embed = None

    def reset_parameters(self):

        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        if self.fc1.bias is not None:
            nn.init.zeros_(self.fc1.bias)

        if self.fc1.bias is not None:
            nn.init.zeros_(self.fc1.bias)

    def set_allow_zero_in_degree(self, set_value):

        self._allow_zero_in_degree = set_value

    def forward(self, graph, feat):

        with graph.local_scope():
            if not self._allow_zero_in_degree:
                if (graph.in_degrees() == 0).any():
                    raise DGLError('There are 0-in-degree nodes in the graph, '
                                   'output for those nodes will be invalid. '
                                   'This is harmful for some applications, '
                                   'causing silent performance regression. '
                                   'Adding self-loop on the input graph by '
                                   'calling `g = dgl.add_self_loop(g)` will resolve '
                                   'the issue. Setting ``allow_zero_in_degree`` '
                                   'to be `True` when constructing this module will '
                                   'suppress the check and let the code run.')

            if self._cached_h is not None:
                feat = self._cached_h
            else:
                # compute normalization
                degs = graph.in_degrees().float().clamp(min=1)
                norm = torch.pow(degs, -0.5)
                norm = norm.to(feat.device).unsqueeze(1)
                # compute (D^-1 A^k D)^k X
                for _ in range(self._k):
                    feat = feat * norm
                    graph.ndata['h'] = feat
                    graph.update_all(fn.copy_u('h', 'm'),
                                     fn.sum('m', 'h'))
                    feat = graph.ndata.pop('h')
                    feat = feat * norm

                # remove feature of a certain index as input of nn.linear
                if self.index_remove is not None:
                    if len(feat) < self.index_remove:
                        raise DGLError('The index to be removed is larger than the number of features')
                    feat = torch.cat((feat[:self.index_remove], feat[self.index_remove + 1:]))

                # normalize the feature
                if self.norm is not None:
                    feat = self.norm(feat)

                # cache feature
                if self._cached:
                    self._cached_h = feat

                x = self.fc1(feat)
                out = self.fc2(x)

                self.embed1 = feat
                self.embed2 = x
                return out

    def num_parameters(self):
        return self.fc1.weight.nelement() + self.fc2.weight.nelement()

    def get_weight(self):
        return self.fc1.weight.data.numpy(), self.fc2.weight.data.numpy()

    def get_embed(self):
        return self.embed1.numpy(), self.embed2.numpy()
