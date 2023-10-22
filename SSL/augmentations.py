
import random
from collections import Counter
import community.community_louvain as community_louvain
from node2vec import Node2Vec
import pandas as pd
import numpy as np
import data.data_functions as data_functions
import time
augmentations_options = ["ID", "NodeDropping", "EdgePerturbation",
                         "AttributeMasking", "Subgraph", "Louvain", "NodeEmbedding"]


def get_augmentation_func(augmentation_name, g):
    if augmentation_name == "ID":
        return ID(g)
    elif augmentation_name == "NodeEmbedding":
        return node_embedding(g)
    elif augmentation_name == "EdgePerturbation":
        return edge_perturbation(g)
    elif augmentation_name == "AttributeMasking":
        return attribute_masking(g)
    elif augmentation_name == "Subgraph":
        return subgraph(g)
    elif augmentation_name == "Louvain":
        return Louvain(g)
    elif augmentation_name == "NodeDropping":
        return node_dropping(g)
    else:
        print("Invalid augmentation {}".format(augmentation_name))
        exit(1)



def ID(G):
    return G


def node_dropping(G, p=0.2):
    """
    Implementation of node dropping augmentation from graphcl
    Args:
        G: a graph
        p: percentage

    Returns: The graph after augmentation

    """
    nodes = G.nodes
    nodes_to_remove = np.random.choice(nodes, size=round(p * len(nodes)), replace=False)
    G.remove_nodes_from(nodes_to_remove)
    G = data_functions.sort_graph_nodes(G)
    return G


def edge_perturbation(G, p=0.1):
    """
    perturb the connectivities in G through randomly adding or dropping
    certain ratio of edges
    Args:
        G: a graph
        p: probability for adding/dropping edge

    Returns: The graph after augmentation

    """
    edges = list(G.edges())
    n_edges = G.number_of_edges()
    n_nodes = G.number_of_nodes()
    # remove edges
    edges_ind_to_remove = np.random.choice(
        list(range(n_edges)), round(p * n_edges), replace=False
    )
    edges_to_remove = [edges[i] for i in edges_ind_to_remove]
    # add edges
    first_node = np.random.choice(list(range(n_nodes)), round(p * n_edges))
    second_node = np.random.choice(list(range(n_nodes)), round(p * n_edges))
    edges_to_add = list(zip(first_node, second_node))
    G.remove_edges_from(edges_to_remove)
    G.add_edges_from(edges_to_add)
    G = data_functions.sort_graph_nodes(G)
    return G


def attribute_masking(G, p=0.2):
    """
    Implementation of attribute masking augmentation from graphcl
    Args:
        G: a networkx graph
        p: percentage

    Returns: The graph after augmentation

    """
    n_nodes = G.number_of_nodes()
    mask_num = int(n_nodes * p)
    nodes_attributes = data_functions.get_attributes_matrix(G)
    token = nodes_attributes.mean(axis=0)
    idx_mask = np.random.choice(n_nodes, mask_num, replace=False)
    # token_mat = np.tile(token, (mask_num, 1))
    nodes_attributes[idx_mask] = token
    G = data_functions.change_attributes(G, nodes_attributes, change_type="replace" )
    return G


def subgraph(G, p=0.2):
    """
    Implementation of attribute subgraph augmentation from graphcl
    Args:
        G: a networkx graph
         p: percentage

    Returns: The graph after augmentation

    """
    node_num = G.number_of_nodes()
    sub_num = int(node_num * (1 - p))
    idx_sub = [np.random.randint(node_num, size=1)[0]]
    idx_neigh = set([n for n in G.neighbors(idx_sub[-1])])

    while len(idx_sub) <= sub_num:
        if len(idx_neigh) == 0:
            idx_unsub = list(set([n for n in range(node_num)]).difference(set(idx_sub)))
            idx_neigh = set([np.random.choice(idx_unsub)])
        sample_node = np.random.choice(list(idx_neigh))

        idx_sub.append(sample_node)
        idx_neigh = idx_neigh.union(
            set([n for n in G.neighbors(idx_sub[-1])])
        ).difference(set(idx_sub))

    idx_nondrop = idx_sub
    idx_nondrop.sort()

    subgraph = data_functions.get_subgraph(G, idx_sub)
    subgraph = data_functions.sort_graph_nodes(subgraph)
    return subgraph


def Louvain(
    G, min_community_size_ratio: float = 0.1, removing_ratio_range=(0.2, 0.55)
):
    start_time = time.time()

    node_num = G.number_of_nodes()
    edge_num = G.number_of_edges()

    # initialize the resolution parameter of Louvain algorithm
    louvain_resolution = 1
    n_tries = 0

    # search for the value of the resolution which leads to an appropriate number of removed nodes
    # higher value causes less separation of the data
    while n_tries < 10:
        n_tries += 1

        partition = community_louvain.best_partition(G, resolution=louvain_resolution)
        threshold = len(G.nodes) * min_community_size_ratio
        counters = dict(Counter(partition.values()).items())
        communities_to_remove = list(
            filter(lambda x: counters.get(x) < threshold, counters.keys())
        )
        nodes_to_remain = [
            k for k in partition.keys() if partition.get(k) not in communities_to_remove
        ]
        nodes_to_remove = list(set(range(G.number_of_nodes())) - set(nodes_to_remain))

        if (
            removing_ratio_range[0]
            <= 1 - len(nodes_to_remain) / G.number_of_nodes()
            <= removing_ratio_range[1]
        ):
            break
        elif 1 - len(nodes_to_remain) / G.number_of_nodes() > removing_ratio_range[1]:
            louvain_resolution += 0.1
        else:
            louvain_resolution -= 0.1

    if 1 - len(nodes_to_remain) / G.number_of_nodes() < removing_ratio_range[0]:
        while 1 - len(nodes_to_remain) / G.number_of_nodes() < removing_ratio_range[0]:
            # print(f'searching... now got {1 - len(nodes_to_remain) / G.number_of_nodes()}, num communities to remove:'
            #       f' {len(communities_to_remove)}')
            communities_to_remove.append(
                min(
                    list(set(counters.keys() - set(communities_to_remove))),
                    key=lambda k: counters[k],
                )
            )
            nodes_to_remain = [
                k
                for k in partition.keys()
                if partition.get(k) not in communities_to_remove
            ]
            nodes_to_remove = list(
                set(range(G.number_of_nodes())) - set(nodes_to_remain)
            )

        # print(f'Found! got {1 - len(nodes_to_remain) / G.number_of_nodes()} - {len(communities_to_remove)} communities'
        #       f' removed')

    if 1 - len(nodes_to_remain) / G.number_of_nodes() > removing_ratio_range[1]:
        nodes_to_add = list(
            random.sample(
                nodes_to_remove,
                int(len(nodes_to_remove) - removing_ratio_range[1] * node_num),
            )
        )
        nodes_to_remain += nodes_to_add
        nodes_to_remove = list(set(nodes_to_remove) - set(nodes_to_add))

    # nodes_to_remove.sort()

    if len(nodes_to_remove) != len(G.nodes):
        G.remove_nodes_from(nodes_to_remove)
        G = data_functions.sort_graph_nodes(G)
    return G


# @timer
def node_embedding(G, n_dimensions: int = 10, dim: int = None):
    """

    Args:
        G: A networkx graph
        ignored_arg:
        n_dimensions:
        dim:

    Returns:

    """
    start_time = time.time()
    assert not dim or dim in range(n_dimensions)

    node_num = G.number_of_nodes()
    edge_num = G.number_of_edges()

    if not dim:
        dim = random.randrange(n_dimensions)
        # print(f'Chosen dimension: {dim}')

    # Generate walks
    node2vec = Node2Vec(G, dimensions=n_dimensions, quiet=True)
    # Learn embedding
    model = node2vec.fit(window=10, min_count=1)

    # get the nodes' embeddings
    node_embeddings = model.wv.vectors

    df_embedding = pd.DataFrame(node_embeddings)

    # center the embeddings
    means = df_embedding.describe().iloc[1]

    for i, col in enumerate(df_embedding.columns):
        df_embedding[col] -= means[i]

    # nodes_to_remain = [i for i in range(df_embedding.shape[0]) if df_embedding[df_embedding.columns[dim]][i] >= 0]
    nodes_to_remove = [
        i
        for i in range(df_embedding.shape[0])
        if df_embedding[df_embedding.columns[dim]][i] < 0
    ]

    G.remove_nodes_from(nodes_to_remove)
    G = data_functions.sort_graph_nodes(G)
    return G


