
import contrastive_learning
from scipy.special import softmax
import pickle
import data.data_functions as data_functions
from augmentations import get_augmentation_func
dataset_list = ["BZR"]
params = {
    "MAX_EPOCHS": 35,
    "lr": 0.01,
    "batch_size": 96,
    "hidden1": 40,
    "hidden2": 35,
    "embedding_size": 50,
    "hidden": 40,
    "loss": "graphcl"
}

augmentations_params = {"NodeDropping": 50,
       "EdgePerturbation": 40,
       "AttributeMasking": 90,
       "Subgraph": 30,
       "Louvain": 60,
       "NodeEmbedding": 70,
       "ID": 10
                            }

p = softmax(
    [augmentations_params["NodeDropping"],
     augmentations_params["EdgePerturbation"],
     augmentations_params["AttributeMasking"],
     augmentations_params["Subgraph"],
     augmentations_params["Louvain"],
     augmentations_params["NodeEmbedding"],
     augmentations_params["ID"]]
)
percentage_dict = {augmentation: p_value for augmentation, p_value in
                   zip(augmentations_params.keys(), p)}
params["percentage_dict"] = percentage_dict

def calculate_all_augmentations(dataset_list):
    augmentations_list = augmentations_params.keys()
    for dataset in dataset_list:
        print(f"Dataset: {dataset}")
        X_train, y_train, X_test, y_test = data_functions.get_data(dataset)
        X_augmented_train = {aug: [] for aug in augmentations_list}
        X_augmented_test = {aug: [] for aug in augmentations_list}
        for augmentation in augmentations_list:
            X_augmented_train[augmentation] = [get_augmentation_func(augmentation, graph.copy()) for graph in X_train]
            X_augmented_test[augmentation] = [get_augmentation_func(augmentation, graph.copy()) for graph in X_test]
        # change the saving format
        for i in range(len(X_augmented_train["ID"])):
            for aug_name in augmentations_list:
                X_augmented_train[(i, aug_name)] = X_augmented_train[aug_name][i]
        for i in range(len(X_augmented_test["ID"])):
            for aug_name in augmentations_list:
                X_augmented_test[(i, aug_name)] = X_augmented_test[aug_name][i]
        pickle.dump({"train": X_augmented_train, "test": X_augmented_test}, open(
            f"Results/{dataset}_augmentations.pkl", "wb"))
    return

def get_augmentations(dataset_name):
    file = open(
        f"Results/{dataset_name}_augmentations.pkl",
        "rb")
    augmentations = pickle.load(file)
    file.close()
    X_augmentations_train = augmentations["train"]
    X_augmentations_test = augmentations["test"]
    return X_augmentations_train, X_augmentations_test
def train_and_get_representation(dataset_name):
    X_train, y_train, X_test, y_test = data_functions.get_data(dataset_name)
    X_augmentations_train, X_augmentations_test = get_augmentations(dataset_name)

    X_train = {i: graph for i, graph in enumerate(X_train)}
    y_train = {i: label for i, label in enumerate(y_train)}
    X_test = {i: graph for i, graph in enumerate(X_test)}
    y_test = {i: label for i, label in enumerate(y_test)}

    model, train_embedding, test_embedding = contrastive_learning.train_and_test \
        (X_train, X_augmentations_train, X_test, X_augmentations_test, params)
    return train_embedding, test_embedding


if __name__ == '__main__':
    # calculate_all_augmentations(dataset_list)
    x1, x2 = train_and_get_representation("BZR")

