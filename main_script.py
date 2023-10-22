import data
import SpectralEmbedding
import SSL.find_contrastive_representation
import SSL.contrastive_and_classification
import MLP
import anomaly_detection
import ToyModel
import numpy as np
datasets_list = ["BZR"]

params = {"representationType": "SpectralEmbedding", "ClassificationOrAnomalyDetection": "Classification"}

if __name__ == '__main__':
    for dataset in datasets_list:
        # Load data
        X_train, y_train, X_test, y_test = data.data_functions.get_data(dataset)

        # if you want to use toy model, uncomment the following line
        # X_train, y_train, X_test, y_test = ToyModel.simulate_toy_model.build_graph_dataset_by_alpha(alpha1=0.1, alpha2=0.4)

        # get representation
        if params["representationType"] == "SpectralEmbedding":
            train_rep, test_rep = SpectralEmbedding.find_best_representation.get_representation(dataset_name=dataset) #Attention: there are many factors, those are the default ones
        elif params["representationType"] == "SSL":
            train_rep, test_rep = SSL.find_contrastive_representation.train_and_get_representation(dataset_name=dataset) #Attention: there are many factors, those are the default ones
        # classify representation
        if params["ClassificationOrAnomalyDetection"] == "Classification":
            model, prediction, scores = MLP.train_and_test(train_rep, y_train, test_rep, y_test)
        elif params["ClassificationOrAnomalyDetection"] == "AnomalyDetection":
            X = np.concatenate((train_rep, test_rep), axis=0)
            y = np.concatenate((y_train, y_test), axis=0)
            anomaly_detection_results = anomaly_detection.grid_clustering(X, y, dataset_name=dataset)


        # if you want to to the contrastive learning and classification together, uncomment the following lines

        # X_augmentations_train, X_augmentations_test = SSL.find_contrastive_representation.get_augmentations(dataset)
        #
        # X_train = {i: graph for i, graph in enumerate(X_train)}
        # y_train = {i: label for i, label in enumerate(y_train)}
        # X_test = {i: graph for i, graph in enumerate(X_test)}
        # y_test = {i: label for i, label in enumerate(y_test)}
        #
        # model, prediction, scores = SSL.contrastive_and_classification.train_and_test(X_train, X_augmentations_train, y_train, X_test, X_augmentations_test, y_test)
