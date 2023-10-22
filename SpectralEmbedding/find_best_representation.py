import numpy as np
import sklearn
import pickle
import represent
from xgboost import XGBClassifier
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
import data.data_functions as data_functions
import pandas as pd
from sklearn.preprocessing import StandardScaler

import os
plt.rcParams.update({'font.size': 14, 'font.family': "Times New Roman"})

k_cross = 5
datasets_list = ["BZR", "NCI109", "reality_mining"]

XGB_params = {
    "gamma": 0.35,
    "max_depth": 6,
    "subsample": 0.85,
    "reg_lambda": 0.1,
    "reg_alpha": 0.25,
    "n_estimators": 200,
    "learning_rate": 0.01,
              }


attributes_type_vec = ["original_attributes_without_topological", "topological_attributes", "all_attributes"]
rep_type_vec = ["attributes_mean", "attributes_matrix_padded", "attributes_svd"]
matrix_type_vec = ["laplacian", "laplacian_norm", "adjacency"]

def calc_and_save_attributes_embedding(datasets_list, attributes_rep_type="mean", attribute_type="all_attributes"):
    for dataset_name in datasets_list:
        print("Dataset {}".format(dataset_name))
        # check if already calculated
        try:
            rep = pd.read_pickle(f"results/Embedding/{attributes_rep_type}/{attribute_type}/{dataset_name}_{attribute_type}_{attributes_rep_type}.pkl")
            print("Already calculated")
            continue
        except:
            pass
        X_train, y_train, X_test, y_test = data_functions.get_data(dataset_name)

        if attributes_rep_type == "attributes_mean":
            rep_train = np.stack([represent.get_mean_attributes(graph, attribute_type) for graph in X_train])
            rep_test = np.stack([represent.get_mean_attributes(graph, attribute_type) for graph in X_test])

        elif attributes_rep_type in ["matrix_padded", "svd"]:
            params = {"n_vertices": 10, "n_eigenvalues": 10, "n_moments": 10, "absolute_value": False,
                      "matrix_type": attribute_type, "rep_type": attributes_rep_type}
            rep_train = [represent.get_eigenvectors_moments(graph, params) for graph in X_train]
            rep_test = [represent.get_eigenvectors_moments(graph, params) for graph in X_test]

            train_moments = np.stack([rep[0][2:, :].flatten() for rep in rep_train])
            test_moments = np.stack([rep[0][2:, :].flatten() for rep in rep_test])

            train_eigenvalues = np.stack([rep[1] for rep in rep_train])
            test_eigenvalues = np.stack([rep[1] for rep in rep_test])

            rep_train = np.concatenate((train_moments, train_eigenvalues), axis=1)
            rep_test = np.concatenate((test_moments, test_eigenvalues), axis=1)

        embedding = {"train": rep_train, "test": rep_test}
        # save representations
        # create directory if not exists
        if not os.path.exists(f"results/Embedding/{attributes_rep_type}/{attribute_type}"):
            os.makedirs(f"results/Embedding/{attributes_rep_type}/{attribute_type}")

        pickle.dump(embedding, open(f"results/Embedding/{attributes_rep_type}/{attribute_type}/{dataset_name}_{attribute_type}_{attributes_rep_type}.pkl", "wb"))

        print("Dataset {} Done".format(dataset_name))
    return

def calc_and_save_embedding_graph_structure(datasets_list, matrix_type="laplacian"):
    for dataset_name in datasets_list:
        print("Dataset {}".format(dataset_name))
        # check if already calculated
        try:
            rep = pd.read_pickle(f"results/Embedding/{matrix_type}/{dataset_name}_{matrix_type}_moments.pkl")
            print("Already calculated")
            continue
        except:
            pass
        X_train, y_train, X_test, y_test = data_functions.get_data(dataset_name)
        params = {"n_vertices": 10, "n_eigenvalues": 10, "n_moments": 10, "absolute_value": False,
                    "matrix_type": matrix_type, "rep_type": "irrelevant"}


        rep_train = [represent.get_eigenvectors_moments(graph, params) for graph in X_train]
        rep_test = [represent.get_eigenvectors_moments(graph, params) for graph in X_test]

        train_moments = np.stack([rep[0][2:, :] for rep in rep_train])
        test_moments = np.stack([rep[0][2:, :] for rep in rep_test])

        train_eigenvalues = np.stack([rep[1] for rep in rep_train])
        test_eigenvalues = np.stack([rep[1] for rep in rep_test])

        rep_train = {"moments": train_moments, "eigenvalues": train_eigenvalues}
        rep_test = {"moments": test_moments, "eigenvalues": test_eigenvalues}

        # rep_train = np.concatenate((train_moments, train_eigenvalues), axis=1)
        # rep_val = np.concatenate((test_moments, test_eigenvalues), axis=1)

        rep = {"train": rep_train, "test": rep_test}
        if not os.path.exists(f"results/Embedding/{matrix_type}"):
            os.makedirs(f"results/Embedding/{matrix_type}")
        pickle.dump(rep, open(f"results/Embedding/{matrix_type}/{dataset_name}_{matrix_type}_moments.pkl", "wb"))
        print("Dataset {} Done".format(dataset_name))
    return

def get_representation(dataset_name, attributes_rep_type="attributes_mean", attribute_type="all_attributes", matrix_type="laplacian", attributesOrLaplacian="both"):
    """
    Args:
        dataset_name: The dataset
        attributes_rep_type: "mean", "matrix_padded", "svd" (relevant only if attributesOrLaplacian == "attributes" or "both")
        attribute_type: "original_attributes_without_topological", "topological_attributes", "all_attributes"  (relevant only if attributesOrLaplacian == "attributes" or "both")
        matrix_type: "laplacian", "laplacian_norm", "adjacency" (relevant only if attributesOrLaplacian == "laplacian" or "both")
        attributesOrLaplacian: Calculate only internal nodes attributes, only graph structure or both

    Returns: The graph representation
    """

    # check if to load the graph structure representation
    if attributesOrLaplacian in ["laplacian", "both"]:
        try:
            rep = pd.read_pickle(f"results/Embedding/{matrix_type}/{dataset_name}_{matrix_type}_moments.pkl")
        except:
            calc_and_save_embedding_graph_structure([dataset_name], matrix_type=matrix_type)
            rep = pd.read_pickle(f"results/Embedding/{matrix_type}/{dataset_name}_{matrix_type}_moments.pkl")
        train_rep_structure = rep["train"]
        test_rep_structure = rep["test"]

        rep_train_structure = np.concatenate((train_rep_structure["moments"].reshape(train_rep_structure["moments"].shape[0], -1), train_rep_structure["eigenvalues"]), axis=1)
        rep_test_structure = np.concatenate((test_rep_structure["moments"].reshape(test_rep_structure["moments"].shape[0], -1), test_rep_structure["eigenvalues"]), axis=1)

        if attributesOrLaplacian == "laplacian":
            return rep_train_structure, rep_test_structure

    if attributesOrLaplacian in ["attributes", "both"]:
        try:
            rep_attributes = pd.read_pickle(f"results/Embedding/{attributes_rep_type}/{attribute_type}/{dataset_name}_{attribute_type}_{attributes_rep_type}.pkl")
        except:
            calc_and_save_attributes_embedding([dataset_name], attributes_rep_type=attributes_rep_type, attribute_type=attribute_type)
            rep_attributes = pd.read_pickle(
                f"results/Embedding/{attributes_rep_type}/{attribute_type}/{dataset_name}_{attribute_type}_{attributes_rep_type}.pkl")
        train_rep_attributes = rep_attributes["train"]
        test_rep_attributes = rep_attributes["test"]

        if attributesOrLaplacian == "attributes":
            train_rep_attributes = np.array(train_rep_attributes)
            test_rep_attributes = np.array(test_rep_attributes)

            train_rep_attributes = np.reshape(train_rep_attributes, (train_rep_attributes.shape[0], -1))
            test_rep_attributes = np.reshape(test_rep_attributes, (test_rep_attributes.shape[0], -1))
            return train_rep_attributes, test_rep_attributes

    if attributesOrLaplacian == "both":
        train_rep = np.concatenate((rep_train_structure, train_rep_attributes), axis=1)
        test_rep = np.concatenate((rep_test_structure, test_rep_attributes), axis=1)
        train_rep = np.reshape(train_rep, (train_rep.shape[0], -1))
        test_rep = np.reshape(test_rep, (test_rep.shape[0], -1))
        return train_rep, test_rep


def train_and_test(datasets_list, attributes_rep_type, attribute_type, matrix_type, attributesOrLaplacian, SAVE_FLAG=False):
    accuracy_all_datasets = pd.DataFrame(columns=["Dataset", "Accuracy_Train", "Accuracy_Validation", "Attribute_Type", "Rep_Type", "Matrix_Type", "attributesOrLaplacian"])
    for dataset_name in datasets_list:
        print(f"Dataset: {dataset_name}")
        X_train, y_train, X_val, y_val = data_functions.get_data(dataset_name)

        # first check if there is a saved representation
        try:
            train_rep, test_rep = get_representation(dataset_name, attributes_rep_type, attribute_type, matrix_type, attributesOrLaplacian)
        except:
            print("No saved representation")
            continue
        X = train_rep
        y = np.array(y_train)
        ind = np.random.permutation(len(y))
        # split ind to k_cross groups
        ind = np.array_split(ind, k_cross)
        # split to groups by k_cross
        X = [X[i] for i in ind]
        y = [y[i] for i in ind]

        for k in range(k_cross):
            X_val = X[k]
            y_val = y[k]
            X_train = np.concatenate([X[i] for i in range(k_cross) if i != k], axis=0)
            y_train = np.concatenate([y[i] for i in range(k_cross) if i != k], axis=0)
            accuracy_train, accuracy_val = train_and_test_model(X_train, y_train, X_val, y_val)
            if not SAVE_FLAG:
                print(f"Train accuracy: {accuracy_train:.3f}, Val accuracy: {accuracy_val:.3f}")
            new_row = {"Dataset": dataset_name, "Accuracy_Train": accuracy_train, "Accuracy_Validation": accuracy_val, "Attribute_Type": attribute_type,
                       "Rep_Type": attributes_rep_type, "Matrix_Type": matrix_type, "attributesOrLaplacian": attributesOrLaplacian}
            accuracy_all_datasets = accuracy_all_datasets.append(new_row, ignore_index=True)

        if SAVE_FLAG:
            accuracy_all_datasets.to_csv(f"results/accuracy_all_datasets.csv", float_format="%.3f")
    return accuracy_all_datasets


def train_and_test_model(X_train, y_train, X_val, y_val):
    # normalize data
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_val = scaler.transform(X_val)
    model = XGBClassifier()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_val)
    accuracy_val = accuracy_score(y_val, y_pred)
    accuracy_train = accuracy_score(y_train, model.predict(X_train))
    return accuracy_train, accuracy_val




def find_eigenvalues_and_eigenvectors_number(dataset_name):
    """ find the influence of the number of eigenvalues and eigenvectors on the accuracy"""
    k_cross = 5

    X_train, y_train, X_val, y_val = data_functions.get_data(dataset_name)

    params = {"n_vertices": 100, "n_eigenvalues": 100,  "n_moments":100, "absolute_value": False,
              "matrix_type": "laplacian_norm", "rep_type": "eigenvectors_moments"}


    rep_train = [represent.get_eigenvectors_moments(graph, params) for graph in X_train]
    rep_val = [represent.get_eigenvectors_moments(graph, params) for graph in X_val]

    train_moments = np.stack([rep[0][2:, :] for rep in rep_train])
    val_moments = np.stack([rep[0][2:, :] for rep in rep_val])

    train_eigenvalues = np.stack([rep[1] for rep in rep_train])
    val_eigenvalues = np.stack([rep[1] for rep in rep_val])

    all_rep_moments = np.concatenate((train_moments, val_moments), axis=0)
    all_rep_eigenvalues = np.concatenate((train_eigenvalues, val_eigenvalues), axis=0)
    labels = np.concatenate((y_train, y_val), axis=0)

    accuracy_eigenvalues = pd.DataFrame([], columns=["Number_of_eigenvalues", "Accuracy_Train", "Accuracy_Val"])
    n_eigenvalues_vec = np.arange(1, 40, 1)
    for n_eigenvalues in n_eigenvalues_vec:
        rep = all_rep_eigenvalues[:, :n_eigenvalues]
        for _ in range(k_cross):
            X_train, X_val, y_train, y_val = sklearn.model_selection.train_test_split(rep, labels)
            # normalize
            scaler = StandardScaler()
            scaler.fit(X_train)
            X_train = scaler.transform(X_train)
            X_val = scaler.transform(X_val)

            model = XGBClassifier()
            model.fit(X_train, y_train)
            y_pred = model.predict(X_val)
            accuracy_val = accuracy_score(y_val, y_pred)
            accuracy_train = accuracy_score(y_train, model.predict(X_train))
            new_row = {"Number_of_eigenvalues": n_eigenvalues, "Accuracy_Train": accuracy_train, "Accuracy_Val":accuracy_val}
            accuracy_eigenvalues = accuracy_eigenvalues.append(new_row, ignore_index=True)
            accuracy_eigenvalues.to_csv(f"accuracy_eigenvalues_{dataset_name}.csv")
    #
    accuracy_eigenvectors_and_moments = pd.DataFrame(columns=["Number_of_moments", "Number_of_eigenvectors", "Accuracy_Train", "Accuracy_Val"])
    n_eigenvectors_vec = np.arange(1, 20, 1)
    n_moments_vec = np.arange(1, 20, 2)
    moments_eigenvectors_grid = np.array(np.meshgrid(n_moments_vec, n_eigenvectors_vec)).T.reshape(-1, 2)

    for n_moments, n_eigenvectors in moments_eigenvectors_grid:
        rep = all_rep_moments[:, :n_moments, :n_eigenvectors].reshape(all_rep_moments.shape[0], -1)
        for _ in range(k_cross):
            X_train, X_val, y_train, y_val = sklearn.model_selection.train_test_split(rep, labels)
            # normalize
            scaler = StandardScaler()
            scaler.fit(X_train)
            X_train = scaler.transform(X_train)
            X_val = scaler.transform(X_val)

            model = XGBClassifier()
            model.fit(X_train, y_train)
            y_pred = model.predict(X_val)
            accuracy_val = accuracy_score(y_val, y_pred)
            accuracy_train = accuracy_score(y_train, model.predict(X_train))
            new_row = {"Number_of_moments": n_moments, "Number_of_eigenvectors": n_eigenvectors, "Accuracy_Train": accuracy_train, "Accuracy_Val":accuracy_val}
            accuracy_eigenvectors_and_moments = accuracy_eigenvectors_and_moments.append(new_row, ignore_index=True)
            accuracy_eigenvectors_and_moments.to_csv(f"accuracy_eigenvectors_and_moments_{dataset_name}.csv")
    return accuracy_eigenvalues, accuracy_eigenvectors_and_moments

# def plot_difference_matrix_type():
#     """ This is an example how to do a t test between the two best performing matrix types, and find which matrix type is the best.
#     We did the same for different attributes types"""
#     matrix_types = ["laplacian_norm", "laplacian", "adjacency"]
#     df = pd.DataFrame(columns=["Dataset", "Matrix_Type", "Accuracy_Train", "Accuracy_Validation"])
#     for matrix_type in matrix_types:
#         results_attribute_type = pd.read_csv(f"results/accuracy_all_datasets_{matrix_type}.csv", index_col=0)
#         results_attribute_type["Matrix_Type"] = matrix_type
#         df = pd.concat([df, results_attribute_type], axis=0)
#
#
#     significant_datasets = []
#     #   do a anova test for each dataset
#     # deletw COLLAB dataset
#     df = df[~df["Dataset"].isin(["COLLAB"])]
#     for dataset_name in df["Dataset"].unique():
#         df_dataset = df[df["Dataset"] == dataset_name]
#         score, pvalue = f_oneway(df_dataset[df_dataset["Matrix_Type"] == "laplacian_norm"]["Accuracy_Validation"],
#                  df_dataset[df_dataset["Matrix_Type"] == "laplacian"]["Accuracy_Validation"],
#                  df_dataset[df_dataset["Matrix_Type"] == "adjacency"]["Accuracy_Validation"])
#         if pvalue < 0.05:
#             #   do a t test between the two best performing
#             accuracy_per_matrix_type = {matrix_type: df_dataset[df_dataset["Matrix_Type"] == matrix_type]["Accuracy_Validation"].mean() for matrix_type in matrix_types}
#             sorted_methods = sorted(accuracy_per_matrix_type, key=accuracy_per_matrix_type.get, reverse=True)
#             # t test between the two best performing
#             score, p_value_t_test = stats.ttest_ind(df_dataset[df_dataset["Matrix_Type"] == sorted_methods[0]]["Accuracy_Validation"],
#                             df_dataset[df_dataset["Matrix_Type"] == sorted_methods[1]]["Accuracy_Validation"])
#             if p_value_t_test < 0.05:
#                 significant_datasets.append(dataset_name)
#
#     #     barplot df
#
#     plt.figure(figsize=(10, 20))
#     g = sns.catplot(kind='bar', data=df, y='Dataset', x='Accuracy_Validation', hue="Matrix_Type")
#     plt.xlabel("Accuracy")
#     plt.ylabel("")
#
#     ylabels = [item.get_text() for item in plt.gca().get_yticklabels()]
#     for dataset in significant_datasets:
#         dataset_ind = ylabels.index(dataset)
#         label = plt.gca().get_yticklabels()[dataset_ind]
#         label.set_bbox(dict(facecolor='none', edgecolor='red'))
#
#     # get legend handles and labels
#
#     handles, labels = plt.gca().get_legend_handles_labels()
#     labels = [" ".join(x.split("_")[:2]) for x in labels]
#     labels = [x.capitalize() for x in labels]
#     # change y labels to upper case
#     ylabels = [" ".join(x.split("_")).upper() for x in ylabels]
#     plt.gca().set_yticklabels(ylabels)
#
#
#     g._legend.set_title("")
#     # replace labels
#     for t, l in zip(g._legend.texts, labels):
#         t.set_text(l)
#     plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
#     plt.tight_layout()
#     # set legend outside
#     plt.savefig(f"results/plots/accuracy_all_datasets_all_matrices_with_ttest_matrixytpe.png")
#     plt.show()



if __name__ == '__main__':
    for attributes_type in attributes_type_vec:
        for rep_type in rep_type_vec:
            for matrix_type in matrix_type_vec:
                get_representation("BZR", attributes_rep_type=rep_type, attribute_type=attributes_type, matrix_type=matrix_type, attributesOrLaplacian="both")









