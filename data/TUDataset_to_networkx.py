import os
import os.path as osp
import shutil
from itertools import repeat
import torch
from torch_geometric.data import InMemoryDataset, download_url, extract_zip, Batch
from torch_geometric.io import read_tu_data
import networkx as nx
from sklearn.preprocessing import OneHotEncoder

from collections import Counter
import pickle
import sys

import numpy as np

datasets_list = ["BZR", "COX2", "ENZYMES", "PROTEINS_full", "NCI1", "PTC_MR", "DD", "NCI109", "COLLAB", "IMDB-BINARY", "IMDB-MULTI"]

# tudataset adopted from torch_geometric==1.1.0
class TUDatasetExt(InMemoryDataset):
    r"""A variety of graph kernel benchmark datasets, *.e.g.* "IMDB-BINARY",
    "REDDIT-BINARY" or "PROTEINS", collected from the `TU Dortmund University
    <http://graphkernels.cs.tu-dortmund.de>`_.

    Args:
        root (string): Root directory where the dataset should be saved.
        name (string): The `name <http://graphkernels.cs.tu-dortmund.de>`_ of
            the dataset.
        transform (callable, optional): A function/transform that takes in an
            :obj:`torch_geometric.data.Data` object and returns a transformed
            version. The data object will be transformed before every access.
            (default: :obj:`None`)
        pre_transform (callable, optional): A function/transform that takes in
            an :obj:`torch_geometric.data.Data` object and returns a
            transformed version. The data object will be transformed before
            being saved to disk. (default: :obj:`None`)
        pre_filter (callable, optional): A function that takes in an
            :obj:`torch_geometric.data.Data` object and returns a boolean
            value, indicating whether the data object should be included in the
            final dataset. (default: :obj:`None`)
        use_node_attr (bool, optional): If :obj:`True`, the dataset will
            contain additional continuous node features (if present).
            (default: :obj:`False`)
    """

    url = "https://ls11-www.cs.uni-dortmund.de/people/morris/" "graphkerneldatasets"

    def __init__(
        self,
        root,
        name,
        transform=None,
        pre_transform=None,
        pre_filter=None,
        use_node_attr=True,
        processed_filename="data.pt",
        dataset=None,
    ):

        self.name = name
        self.processed_filename = processed_filename
        super(TUDatasetExt, self).__init__(root, transform, pre_transform, pre_filter)

        if dataset is None:
            self.data, self.slices = torch.load(self.processed_paths[0])

        else:
            self.data, self.slices = dataset.data, dataset.slices

        if self.data.x is not None and not use_node_attr:
            self.data.x = self.data.x[:, self.num_node_attributes :]

        self.node_labels = get_node_labels(name)

    @property
    def num_node_labels(self):
        if self.data.x is None:
            return 0
        for i in range(self.data.x.size(1)):
            if self.data.x[:, i:].sum().item() == self.data.x.size(0):
                return self.data.x.size(1) - i
        return 0

    @property
    def num_node_attributes(self):
        if self.data.x is None:
            return 0
        return self.data.x.size(1) - self.num_node_labels

    @property
    def raw_file_names(self):
        # names = ['A']
        names = ["A", "graph_indicator"]
        return ["{}_{}.txt".format(self.name, name) for name in names]
        # return [f'{name}.txt' for name in names]

    @property
    def processed_file_names(self):
        return self.processed_filename

    @property
    def num_node_features(self):
        r"""Returns the number of features per node in the dataset."""
        return self[0][0].num_node_features

    def download(self):
        path = download_url("{}/{}.zip".format(self.url, self.name), self.root)
        extract_zip(path, self.root)
        os.unlink(path)
        shutil.rmtree(self.raw_dir)
        os.rename(osp.join(self.root, self.name), self.raw_dir)

    def process(self):
        self.data, self.slices, self.sizes = read_tu_data(self.raw_dir, self.name)

        if self.pre_filter is not None:
            data_list = [self.get(idx)[0] for idx in range(len(self))]
            data_list = [data for data in data_list if self.pre_filter(data)]
            self.data, self.slices = self.collate(data_list)

        if self.pre_transform is not None:
            data_list = [self.get(idx)[0] for idx in range(len(self))]
            data_list = [self.pre_transform(data) for data in data_list]
            self.data, self.slices = self.collate(data_list)

        torch.save((self.data, self.slices), self.processed_paths[0])

    def __repr__(self):
        return "{}({})".format(self.name, len(self))

    def get(self, idx):
        data = self.data.__class__()
        if hasattr(self.data, "__num_nodes__"):
            data.num_nodes = self.data.__num_nodes__[idx]

        for key in self.data.keys:
            if key in self.slices.keys():
                item, slices = self.data[key], self.slices[key]
                if torch.is_tensor(item):
                    s = list(repeat(slice(None), item.dim()))
                    s[self.data.__cat_dim__(key, item)] = slice(
                        slices[idx], slices[idx + 1]
                    )
                else:
                    s = slice(slices[idx], slices[idx + 1])
                data[key] = item[s]
        if self.node_labels is None:
            data["node_labels"] = None
        else:
            s = slice(slices[idx], slices[idx + 1])
            data["node_labels"] = self.node_labels[s]

        return data


def custom_collate(data_list):
    batch = Batch.from_data_list(
        [d[0] for d in data_list], follow_batch=["edge_index", "edge_index_neg"]
    )
    batch_1 = Batch.from_data_list([d[1] for d in data_list])
    batch_2 = Batch.from_data_list([d[2] for d in data_list])
    return batch, batch_1, batch_2


class DataLoader(torch.utils.data.DataLoader):
    def __init__(self, dataset, batch_size=1, shuffle=True, **kwargs):
        super(DataLoader, self).__init__(
            dataset, batch_size, shuffle, collate_fn=custom_collate, **kwargs
        )


def dataset_to_networkx(data, real_labels=False):
    labels = []
    graphs = []
    attr_shapes = []
    n_nodes = []
    for graph in data:
        label = graph.y.item()
        labels.append(label)
        attribute_matrix = graph.x
        attr_shapes.append(graph.num_features)
        n_nodes.append(graph.num_nodes)
        edges = graph.edge_index
        # node_labels = graph["node_labels"]
        node_labels = None
        graphs.append(create_networkx_graph(edges, attribute_matrix, node_labels))
    # The smaller class is anomaly, the larger class is normal
    # find the larger label
    # anomaly_label = min(labels, key=labels.count)
    # if not real_labels:
    #     labels = [1 if label == anomaly_label else 0 for label in labels]
    return {"graphs": graphs, "labels": labels}


def create_networkx_graph(edges, attributes, node_labels):
    G = nx.from_edgelist(edges.t().tolist())
    if node_labels is not None:
        node_labels = torch.tensor(transformer.transform(node_labels).toarray())
        if attributes is not None:
            attributes = torch.concat((attributes, node_labels), axis=1)

    if attributes is not None:
        for i in range(attributes.size(0)):
            G.add_node(i, attr_dict=attributes[i, :].tolist())
    else:
        for i in range(len(G.nodes)):
            G.add_node(i, attr_dict=[])
    return G

def get_node_labels(dataset_name):
    try:
        f = open(f"TU_DATASETS/{dataset_name}/raw/{dataset_name}_node_labels.txt", "r")
        data = f.read().split("\n")
        data = np.array(list(map(int, data[:-1]))).reshape(-1, 1)
        f.close()
        encoder = OneHotEncoder()
        global transformer
        transformer = encoder.fit(data)
    except:
        print("No node labels for dataset {}".format(dataset_name))
        data=None
    return data


if __name__ == "__main__":
    for dataset_name in datasets_list:
        data = TUDatasetExt(
            root="TU_DATASETS/{}".format(dataset_name),
            name="{}".format(dataset_name),
            use_node_attr=True,
        )
        dataset = dataset_to_networkx(data, real_labels=True)
        # check if there is a directory
        if not os.path.exists("pickle_datasets"):
            os.mkdir("pickle_datasets")
        file = open("pickle_datasets/{}.pkl".format(dataset_name), "wb")
        pickle.dump(dataset, file)
        file.close()
        print("Dataset {} saved".format(dataset_name))
