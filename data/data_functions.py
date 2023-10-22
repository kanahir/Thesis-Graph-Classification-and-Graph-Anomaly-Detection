import pickle
from sklearn.model_selection import train_test_split
import networkx as nx
import os
import torch
import numpy as np
from graphMeasures import FeatureCalculator
import pathlib

path = os.getcwd()


# current_path = str(pathlib.Path().resolve())
# path_dir = current_path.split("/")
# thesis_ind = int(np.where(["Thesis" in x for x in path_dir])[0])
# data_path = "/".join(path_dir[:thesis_ind+1])

feats = ["average_neighbor_degree",
         "degree",
         "k_core",
         "betweenness_centrality",
         "eccentricity",
         "motif3",
         "attractor_basin",
         "load_centrality"]

datasets_list = [
                 "BZR"
                 ]


def split_datasets(datasets_list):
    def split_dataset(dataset):
        graphs = dataset['graphs']
        labels = dataset['labels']
        # get validation set
        graphs_train, graphs_test, labels_train, labels_test = train_test_split(graphs, labels, stratify=labels,
                                                                              test_size=0.2,)
        return {
            "train": {"graphs": graphs_train, "labels": labels_train},
             "test": {"graphs": graphs_test, "labels": labels_test}
        }

    for dataset_name in datasets_list:
        dataset = pickle.load(open("pickle_normalized_datasets/{}.pkl".format(dataset_name), 'rb'))
        dataset_split = split_dataset(dataset)
        if not os.path.exists("split_datasets"):
            os.mkdir("split_datasets")
        pickle.dump(dataset_split, open("split_datasets/{}_split.pkl".format(dataset_name), 'wb'))
        print("Dataset {} Saved".format(dataset_name))


def get_data(dataset_name):
    """
    return the dataset as a list of networkx graphs
    """
    dataset_path = f"data/split_datasets/{dataset_name}_split.pkl"
    dataset = read_dataset(dataset_path)
    X_train = dataset["train"]["graphs"]
    y_train = dataset["train"]["labels"]
    X_test = dataset["test"]["graphs"]
    y_test = dataset["test"]["labels"]
    y_train = np.array(y_train)
    y_test = np.array(y_test)
    return X_train, y_train, X_test, y_test




def add_self_loops(G):
    edges = [(n, n) for n in G.nodes]
    G.add_edges_from(edges)
    return G
def get_feats():
    return feats



def get_attributes(G, returnAsTensor=True):
    attr = nx.get_node_attributes(G, "attr_dict")
    if returnAsTensor:
        return torch.tensor(np.array(list(attr.values())))
    else:
        return np.array(list(attr.values()))

def get_edges(G, returnAsTensor=True):
    edges_list = list(G.edges)
    first_node = list(map(lambda x: x[0], edges_list))
    second_node = list(map(lambda x: x[1], edges_list))
    if returnAsTensor:
        return torch.tensor(np.array([first_node, second_node]))
    else:
        return np.array([first_node, second_node])

def graph_to_input(graph, returnAsTensor=True):
    """
    The function get a networkx graph and return it as format as model input
    """
    G = add_self_loops(graph)
    G_attr, g_edge = get_attributes(G), get_edges(G)
    if returnAsTensor:
        return torch.tensor(G_attr), torch.tensor(g_edge)
    else:
        return G_attr, g_edge



def read_dataset(dataset_path):
    with open(dataset_path, "rb") as file:
        dataset = pickle.load(file)
    return dataset


def sort_graph_nodes(graph):
    nodes_attributes = list(graph.nodes(data=True))
    edges = graph.edges
    nodes_indexing = {node: i for i, node in enumerate(graph.nodes)}
    nodes = list(map(lambda x: nodes_indexing[x], graph.nodes))
    nodes = sorted(nodes)
    new_nodes_attributes = [(nodes_indexing[node], node_dict) for node, node_dict in nodes_attributes]
    e = []
    for edge in edges:
        e.append((nodes_indexing[edge[0]], nodes_indexing[edge[1]], 1))
    g = nx.Graph() # this line was added to restart the graph definition
    g.add_nodes_from(new_nodes_attributes)
    g.add_weighted_edges_from(e)
    return g


def change_attributes(graph, nodes_attributes, change_type="concat"):
    edges = graph.edges
    g = nx.Graph()  # this line was added to restart the graph definition
    topological_attributes = [node_dict.tolist() for node, node_dict in zip(graph.nodes, nodes_attributes)]
    # concat with the previous attributes
    if change_type == "concat":
        A = get_attributes_matrix(graph)
        A_list = [A[node, :].tolist() for node in graph.nodes]
        new_nodes_attributes = [(node, {'attr_dict': np.concatenate([node_topo, node_atrr]).tolist()}) for node, node_topo, node_atrr in zip(graph.nodes, topological_attributes, A_list)]
    elif change_type == "replace":
        new_nodes_attributes = [(node, {'attr_dict': node_topo}) for node, node_topo in zip(graph.nodes, topological_attributes)]
    else:
        raise ValueError("change_type must be 'concat' or 'replace'")
    g.add_nodes_from(new_nodes_attributes)
    g.add_edges_from(edges)
    return g


def get_attributes_matrix(G, attribute_type="all_attributes"):
    """
    Args:
        G: a networkx graph. The topological features are the first features

    Returns: attribute matrix, shape = (n_nodes, n_attributes)

    """
    nodes_attributes = G.nodes(data=True)
    try:
        attributes = np.array([attr[1]['attr_dict'] for attr in nodes_attributes])
    except:
        attributes = np.zeros([len(G.nodes), 0])
    n_feats = len(feats)
    if attribute_type == "all_attributes":
        return attributes
    elif attribute_type == "topological_attributes":
        return attributes[:, :n_feats]
    elif attribute_type == "original_attributes_without_topological":
        return attributes[:, n_feats:]
    else:
        raise ValueError("attribute_type should be one of the following: all, topological, original_attributes_without_topological")



def calc_topological_features(graph, dataset_name, i=-1):

    ftr_calc = FeatureCalculator(graph, feats, dir_path=f"graph_{i}_{dataset_name}", acc=False, directed=False, gpu=False,
                                 device=0,
                                 verbose=True,
                                 should_zscore=False)
    # calculate the features. If one do not want the features to be saved,
    # one should set the parameter 'should_dump' to False (set to True by default).

    ftr_calc.calculate_features(should_dump=False)
    mx = ftr_calc.get_features()

    # Ignore nan columns
    mx = mx.dropna(axis=1, how='all').values

    graph = change_attributes(graph, mx)
    return graph



def add_topological_features(datasets_list):
    for dataset_name in datasets_list:
        print("Starting Dataset {}".format(dataset_name))
        dataset = pickle.load(open("pickle_datasets/{}.pkl".format(dataset_name), 'rb'))
        graphs = dataset['graphs']
        graphs = [sort_graph_nodes(g) for g in graphs]
        labels = dataset['labels']

        graphs_topo = [calc_topological_features(g, dataset_name, i) for i, g in enumerate(graphs)]

        new_dataset = {"graphs": graphs_topo,
                               "labels": labels}
        if not os.path.exists("pickle_normalized_datasets"):
            os.mkdir("pickle_normalized_datasets")
        pickle.dump(new_dataset, open("pickle_normalized_datasets/{}.pkl".format(dataset_name), 'wb'))
        print("Dataset {} Saved".format(dataset_name))



def extract_topological_features(X):
    """
    Get a matrix off all graph attributes and extract only the topological features (the first 8 features)
    """
    n_feats = len(feats)
    X_attributes = [get_attributes_matrix(G) for G in X]

    num_nodes = np.array([attr.shape[0] for attr in X_attributes])
    # max_node_allowed = int(np.percentile(num_nodes, 75))
    max_node_allowed = np.max(num_nodes)

    # the topological features are the first
    X_attributes = [attr[:max_node_allowed, :n_feats] for attr in X_attributes]

    return X_attributes


def replace_attributes_to_attributes_norm(graph_dataset):
    X_attributes_orig = [get_attributes_matrix(G) for G in graph_dataset]
    X_attributes = normalize_features(X_attributes_orig)
#     replace attributes
    new_graph_dataset = [change_attributes(G, X_attributes[i], change_type="replace") for i, G in enumerate(graph_dataset)]
    return new_graph_dataset


def normalize_features(X_attributes):
    """
     The function normalize the features, for the topological features some should be log normalized and some zscore.
     The original attributes should be zscore normalized
     Args:
         X_attributes:

     Returns:

     """
    for i in range(X_attributes[0].shape[1]):
        feature_all_values = np.concatenate([x[:, i] for x in X_attributes])
        if i < len(feats) and feats[i] in ["k_core", "motif3", "attractor_basin"]:
            for x in X_attributes: # log
                x[:, i] = np.log(x[:, i])
                #     replace inf with max value
                x[:, i] = np.where(x[:, i] == -np.inf, np.max(x[:, i]), x[:, i])
                x[:, i] = np.nan_to_num(x[:, i], nan=0, posinf=0, neginf=0)
        else: # z score
            for x in X_attributes:
                mean = np.mean(feature_all_values)
                std = np.std(feature_all_values)
                x[:, i] = (x[:, i] - mean) / std
                x[:, i] = np.nan_to_num(x[:, i], nan=0, posinf=0, neginf=0)
    return X_attributes


def get_subgraph(G, nodes):
    """
    The function extract subgraph from the whole graph
    Args:
        G: a networkx graph
        nodes: nodes in the subgraph

    Returns: a subgraph

    """
    subgraph_edges = [edge for edge in G.edges() if edge[0] in nodes and edge[1] in nodes]
    all_nodes_attributes = G.nodes(data=True)
    subgraph_nodes_attributes = [(node, node_dict) for node, node_dict in all_nodes_attributes if node in nodes]
    g = nx.Graph()  # this line was added to restart the graph definition
    g.add_nodes_from(subgraph_nodes_attributes)
    g.add_edges_from(subgraph_edges)
    g = sort_graph_nodes(g)
    return g

if __name__ == '__main__':
    # calculate topology features
    add_topological_features(datasets_list)
    # normalize the attributes
    for dataset_name in datasets_list:
        dataset = pickle.load(open("pickle_datasets/{}.pkl".format(dataset_name), 'rb'))
        graphs = dataset["graphs"]
        normalized_graphs = replace_attributes_to_attributes_norm(graphs)
        dataset["graphs"] = normalized_graphs
        pickle.dump(dataset, open("pickle_normalized_datasets/{}.pkl".format(dataset_name), 'wb'))
    # split the datasets to train and test
    split_datasets(datasets_list)



