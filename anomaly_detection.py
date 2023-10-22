import pandas as pd
from sklearn.mixture import GaussianMixture


from data.data_functions import *
from sklearn.neighbors import LocalOutlierFactor
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
import hdbscan
from sklearn.metrics import precision_score, recall_score, accuracy_score

method = "graph2vec"
datasets_list = ["AIDS", "BZR", "COX2", "ENZYMES",
 "enron", "reality_mining", "sec_repo", "twitter_security"]


clustering_methods_list = ["GMM", "LocalOutlier", "IsolationForest", "SVM", "Random", "HDBSCAN"]

anomaly_label = 1

def GMM(X, n_anomalies, k=5):
    pred = np.zeros(X.shape[0])
    gmm_cluster = GaussianMixture(n_components=k).fit(X)
    # Predict the class based on the max class in the cluster
    scores = gmm_cluster.score_samples(X)
    # find anomalies
    sort_index = np.argsort(scores)
    pred[sort_index[:n_anomalies]] = anomaly_label
    return pred


def localOutlier(X, n_anomalies, n_neighbors=10):
    pred = np.zeros(X.shape[0])
    clf = LocalOutlierFactor(n_neighbors=n_neighbors, novelty=True).fit(X)
    scores = clf.negative_outlier_factor_
    # find anomalies
    sort_index = np.argsort(scores)
    pred[sort_index[:n_anomalies]] = anomaly_label
    return pred


def IsolationForestClf(X, n_anomalies):
    pred = np.zeros(X.shape[0])
    clf = IsolationForest(random_state=0).fit(X)
    scores = clf.score_samples(X)
    # find anomalies
    sort_index = np.argsort(scores)
    pred[sort_index[:n_anomalies]] = anomaly_label
    return pred


def oneClassSvmClf(X, n_anomalies):
    pred = np.zeros(X.shape[0])
    clf = OneClassSVM(gamma='auto').fit(X)
    scores = clf.score_samples(X)
    # find anomalies
    sort_index = np.argsort(scores)
    pred[sort_index[:n_anomalies]] = anomaly_label
    return pred


def Random_score(X, n_anomalies):
    pred = np.zeros(X.shape[0])
    random_indexes = np.random.choice(list(range(len(pred))), n_anomalies, replace=False)
    pred[random_indexes] = anomaly_label
    return pred


def HDBSCAN(X, n_anomalies):
    pred = np.zeros(X.shape[0])
    clf = hdbscan.HDBSCAN().fit(X)
    scores = clf.probabilities_
    # find anomalies
    sort_index = np.argsort(scores)
    pred[sort_index[:n_anomalies]] = anomaly_label
    return pred


def calc_scores(X, y, clustering_method="GMM"):
    anomalies_num = sum(y == anomaly_label)
    if clustering_method == "GMM":
        y_pred = GMM(X, anomalies_num)
    elif clustering_method == "LocalOutlier":
        y_pred = localOutlier(X, anomalies_num)
    elif clustering_method == "IsolationForest":
        y_pred = IsolationForestClf(X, anomalies_num)
    elif clustering_method == "SVM":
        y_pred = oneClassSvmClf(X, anomalies_num)
    elif clustering_method == "Random":
        y_pred = Random_score(X, anomalies_num)
    elif clustering_method == "HDBSCAN":
        y_pred = HDBSCAN(X, anomalies_num)
    else:
        print("Invalid classification method, don't recognize {} method".format(clustering_method))
        exit()
    return precision_score(y, y_pred), recall_score(y, y_pred), accuracy_score(y, y_pred)



def grid_clustering(X, y, dataset_name):
    results = pd.DataFrame(columns=["Dataset", "Clustering-method", "Precision", "Recall", "Accuracy"])
    for clustering_method in clustering_methods_list:
        precision, recall, accuracy = calc_scores(X, y, clustering_method)
        new_raw = {"Dataset": dataset_name, "Clustering-method": clustering_method, "Precision": precision, "Recall": recall, "Accuracy": accuracy}
        results = results.append(new_raw, ignore_index=True)
        print("Clustering method: {}, precision: {:.2f}, recall: {:.2f}".format(clustering_method, precision, recall))
    return results




if __name__ == '__main__':
    pass
    # anomaly_detection_df = pd.DataFrame(columns=["Dataset", "Clustering-method", "Precision", "Recall", "Accuracy"])
    # for dataset_name in :
    #     clustering_results = grid_clustering(rep_all, y_all, dataset_name)
    #     anomaly_detection_df = anomaly_detection_df.append(clustering_results)
    #     anomaly_detection_df.to_csv("anomaly_detection.csv")


