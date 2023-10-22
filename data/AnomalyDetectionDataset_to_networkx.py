import networkx as nx
import numpy as np
import pickle
import os
# This script converts a dataset to a networkx graph
# The dataset is enron/reality_mining/sec_repo/twitter_security

datasets_list = ["enron", "reality_mining", "sec_repo", "twitter_security"]

def dataset_to_networkx(data):
    graph_valid = data['_graph_valid']
    anomaly_index = np.where(np.array(list(graph_valid.values())) == False)[0]
    # give 1 label to normal graph and 0 to anomaly graph
    labels = [1 if i in anomaly_index else 0 for i in range(len(graph_valid))]
    graph_names = list(graph_valid.keys())
    graphs = [graph_to_networkx(data["_node_lists"][i], data["_source"][i]) for i in graph_names]
    return {"graphs": graphs, "labels": labels}


def graph_to_networkx(nodes, edges):
    nodes_dict = {node: i for i, node in enumerate(nodes)}
    n_nodes = len(nodes)
    G = nx.Graph()
    for node in range(n_nodes):
        G.add_node(node)
    for edge in edges:
        G.add_edge(nodes_dict[edge[0]], nodes_dict[edge[1]])
    return G


if __name__ == '__main__':
    for dataset_name in datasets_list:
        data = pickle.load(open(f"TPGAD_DATASETS/{dataset_name}", "rb"))
        dataset = dataset_to_networkx(data)
        # check if there is a directory
        if not os.path.exists("pickle_datasets"):
            os.mkdir("pickle_datasets")
        file = open("pickle_datasets/{}.pkl".format(dataset_name), "wb")
        pickle.dump(dataset, file)
        file.close()
        print("Dataset {} saved".format(dataset_name))