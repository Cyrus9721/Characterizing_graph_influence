import dgl
from dgl.data import CoraGraphDataset, CiteseerGraphDataset, PubmedGraphDataset, RedditDataset


# load the graph dataset
def load_graph_dataset(name):
    if name == 'cora':
        dataset = CoraGraphDataset()
    elif name == 'citeseer':
        dataset = CiteseerGraphDataset()
    elif name == 'pubmed':
        dataset = PubmedGraphDataset()
    elif name == 'reddit':
        dataset = RedditDataset()
    else:
        raise ValueError('Unknown dataset')
    number_classes = dataset.num_classes
    graph = dataset[0]
    graph = dgl.remove_self_loop(graph)
    graph = dgl.add_self_loop(graph)

    train_mask = graph.ndata.pop('train_mask')
    val_mask = graph.ndata.pop('val_mask')
    test_mask = graph.ndata.pop('test_mask')

    feat = graph.ndata.pop('feat')
    labels = graph.ndata.pop('label')

    return graph, feat, labels, train_mask, val_mask, test_mask, number_classes


def main():
    graph, feat, labels, train_mask, val_mask, test_mask, num_labels = load_graph_dataset('cora')
    print(feat.shape)


if __name__ == '__main__':
    main()
