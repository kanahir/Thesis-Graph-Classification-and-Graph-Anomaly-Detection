import random
import numpy as np
import pandas as pd
import networkx as nx

alpha = 0.9
n_nodes = 100
n_graphs = 500
p0 = 10/n_nodes


def create_graph(params):
    """
    :param ind: [class_1_ind, class_2_ind]
    :param class2_ind:
    :return:
    """
    alpha_prob = params["alpha_prob"]
    prob_to_edge = params["prob_to_edge"]
    n_nodes = params["n_vertices"]

    ind = list(range(n_nodes))
    random.shuffle(ind)
    class1ind = ind[:round(n_nodes/2)]
    class2ind = ind[round(n_nodes/2):]
    labels = [0 if ind in class1ind else 1 for ind in range(n_nodes)]

    edge_possibility_same_class = alpha_prob * prob_to_edge
    edge_possibility_diff_class = (1-alpha_prob) * prob_to_edge

    first_node = []
    second_node = []

    for node in class1ind:
        class_1 = np.random.uniform(0, 1, len(class1ind))
        has_edge = np.where(class_1 < edge_possibility_same_class)[0]
        first_node = first_node + [node]*len(has_edge)
        second_node = second_node + list(has_edge)
        # add self loop
        first_node = first_node + [node]
        second_node = second_node + [node]

        class_2 = np.random.uniform(0, 1, len(class2ind))
        has_edge = np.where(class_2 < edge_possibility_diff_class)[0] + len(class_1)
        first_node = first_node + [node]*len(has_edge)
        second_node = second_node + list(has_edge)

    for node in class2ind:
        class_1 = np.random.uniform(0, 1, len(class1ind))
        has_edge = np.where(class_1 < edge_possibility_diff_class)[0]
        first_node = first_node + [node]*len(has_edge)
        second_node = second_node + list(has_edge)
        # add self loop
        first_node = first_node + [node]
        second_node = second_node + [node]

        # class_2 = np.random.uniform(0, 1, len(class2ind))
        # has_edge = np.where(class_2 < edge_possibility_same_class)[0] + len(class_1)
        # first_node = first_node + [node]*len(has_edge)
        # second_node = second_node + list(has_edge)

    square_edges = pd.DataFrame({"source": first_node, "target": second_node})
    g = nx.from_pandas_edgelist(square_edges)
    if g.number_of_nodes() != n_nodes:
        a=1
    return g, labels


def create_graph_by_density_or_groupsize(params):
    """
    :param
    :param class2_ind:
    :return:
    """
    ratio_density = params["ratio_density"]
    n_nodes = params["n_vertices"]
    ratio_between_groups = params["ratio_between_groups"]

    density_group1 = 1/10
    density_group2 = density_group1/ratio_density

    n_vertices_group1 = round(n_nodes*ratio_between_groups/(1+ratio_between_groups))

    ind = list(range(n_nodes))
    random.shuffle(ind)
    class1ind = ind[:n_vertices_group1]
    class2ind = ind[n_vertices_group1:]
    labels = [0 if ind in class1ind else 1 for ind in range(n_nodes)]


    first_node = []
    second_node = []

    for node in class1ind:
        class_1 = np.random.uniform(0, 1, len(class1ind))
        has_edge = np.where(class_1 < density_group1)[0]
        first_node = first_node + [node]*len(has_edge)
        second_node = second_node + list(has_edge)
        # add self loop
        first_node = first_node + [node]
        second_node = second_node + [node]


    for node in class2ind:
        class_1 = np.random.uniform(0, 1, len(class1ind))
        has_edge = np.where(class_1 < density_group2)[0]
        first_node = first_node + [node]*len(has_edge)
        second_node = second_node + list(has_edge)
        # add self loop
        first_node = first_node + [node]
        second_node = second_node + [node]


    # check if the ratio is correct
    # edges_in_group1 = len([i for i in first_node if i in class1ind])
    # edges_in_group2 = len([i for i in first_node if i in class2ind])
    # if edges_in_group1 ==0:
    #     density_group1_actual = density_group1
    # elif edges_in_group2 ==0:
    #     density_group2_actual = density_group2
    # else:
    #     density_group1_actual = edges_in_group1/(len(class1ind)**2)
    #     density_group2_actual = edges_in_group2/(len(class2ind)**2)
    # if abs(density_group1_actual - density_group1) > 0.01 or abs(density_group2_actual - density_group2) > 0.01:
    #     print("density is not correct")
    square_edges = pd.DataFrame({"source": first_node, "target": second_node})
    g = nx.from_pandas_edgelist(square_edges)
    return g, labels

def build_graph_dataset_by_alpha(alpha1, alpha2):
    graph_dataset1 = [create_graph({"alpha_prob": alpha1, "prob_to_edge": p0, "n_vertices": n_nodes})[0] for _ in range(round(n_graphs/2))]
    graph_dataset2 = [create_graph({"alpha_prob": alpha2, "prob_to_edge": p0, "n_vertices": n_nodes})[0] for _ in range(round(n_graphs / 2))]
    graph_dataset = graph_dataset1 + graph_dataset2
    labels = np.array([0]*round(n_graphs/2) + [1]*round(n_graphs/2))

    # shuffle
    indices = np.array(list(range(n_graphs)))
    random.shuffle(indices)
    graph_dataset = [graph_dataset[i] for i in indices]
    labels = labels[indices]
    return graph_dataset, labels


def build_graph_dataset_by_density(ratio_density1, ratio_density2):
    graph_dataset1 = [create_graph_by_density_or_groupsize({"ratio_density": ratio_density1, "ratio_between_groups": 1, "n_vertices": n_nodes})[0] for _ in range(round(n_graphs/2))]
    graph_dataset2 = [create_graph_by_density_or_groupsize({"ratio_density": ratio_density2, "ratio_between_groups": 1, "n_vertices": n_nodes})[0] for _ in range(round(n_graphs / 2))]
    graph_dataset = graph_dataset1 + graph_dataset2
    labels = np.array([0]*round(n_graphs/2) + [1]*round(n_graphs/2))

    # shuffle
    indices = np.array(list(range(n_graphs)))
    random.shuffle(indices)
    graph_dataset = [graph_dataset[i] for i in indices]
    labels = labels[indices]
    return graph_dataset, labels


def build_graph_dataset_by_group_size(ratio_between_groups1, ratio_between_groups2):
    graph_dataset1 = [create_graph_by_density_or_groupsize({"ratio_density": 1, "ratio_between_groups": ratio_between_groups1, "n_vertices": n_nodes})[0] for _ in range(round(n_graphs/2))]
    graph_dataset2 = [create_graph_by_density_or_groupsize({"ratio_density": 1, "ratio_between_groups": ratio_between_groups2, "n_vertices": n_nodes})[0] for _ in range(round(n_graphs / 2))]
    graph_dataset = graph_dataset1 + graph_dataset2
    labels = np.array([0]*round(n_graphs/2) + [1]*round(n_graphs/2))

    # shuffle
    indices = np.array(list(range(n_graphs)))
    random.shuffle(indices)
    graph_dataset = [graph_dataset[i] for i in indices]
    labels = labels[indices]
    return graph_dataset, labels




if __name__ == '__main__':
    graph_dataset, labels = build_graph_dataset_by_alpha(alpha1=0.1, alpha2=0.4)




