import random
import numpy as np
import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, recall_score
from sklearn.preprocessing import OneHotEncoder
import data_functions
import time

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

params = {
       "MAX_EPOCHS": 35,
       "lr": 0.01,
       "batch_size": 96,
       "hidden1": 40,
       "hidden2": 35,
       "embedding_size": 50,
       "hidden": 40}
augmentations_params =    {
    "NodeDropping": 10,
       "EdgePerturbation": 10,
       "AttributeMasking": 10,
       "Subgraph": 10,
       "Louvain": 10,
       "NodeEmbedding": 10,
       "ID": 10,

}
percentage_dict = {augmentation: p_value for augmentation, p_value in  zip(augmentations_params.keys(), list(augmentations_params.values()))}
params["percentage_dict"] = percentage_dict
class AugmentedGraphDataset(Dataset):
    def __init__(self, graphs, graph_augmentations, labels, p, device=device):
        graphs_index = list(graphs.keys())
        self.get_index_dict = {new_index: graph_index for new_index, graph_index in enumerate(graphs_index)}
        self.device = device
        self.probability_dict = p
        # add self loops
        self.graphs = {i: data_functions.add_self_loops(g) for i, g in graphs.items()}
        self.graph_augmentations = graph_augmentations
        #   labels to onehot
        enc = OneHotEncoder(handle_unknown="ignore")
        self.labels = labels
        labels_for_transform = np.array(list(labels.values()))
        self.labels_encoder = enc.fit(labels_for_transform.reshape(-1, 1))

    def __len__(self):
        return len(self.graphs)

    def __getitem__(self, idx):
        graph_ind = self.get_index_dict[idx]
        aug1, aug2 = self.get_augmentations(graph_ind, self.probability_dict)
        attr1 = data_functions.get_attributes(aug1).to(self.device)
        edges1 = data_functions.get_edges(aug1)
        attr2 = data_functions.get_attributes(aug2).to(self.device)
        edges2 = data_functions.get_edges(aug2)
        return (attr1, edges1), (attr2, edges2), torch.tensor(self.labels_encoder.transform(np.array(self.labels[graph_ind]).reshape(-1,1)).todense())

    def return_data(self):
        graphs = []
        labels = []
        for i in range(len(self)):
            graph_ind = self.get_index_dict[i]
            graph = self.graph_augmentations[(graph_ind, "ID")]
            graph = data_functions.add_self_loops(graph)
            attr = data_functions.get_attributes(graph).to(self.device)
            edges = data_functions.get_edges(graph)
            graphs.append((attr, edges))
            labels.append(self.labels[graph_ind])
        return graphs, labels

    def attributes_size(self):
        g = list(self.graphs.values())[0]
        attr = data_functions.get_attributes(g)
        if len(attr.shape) > 2:
            attr = attr.squeeze()
        return int(attr.shape[1])

    def get_augmentations(self, graph_ind, p):
        options_aug = list(p.keys())
        values = list(p.values())
        augmentations = random.choices(options_aug, weights=values, k=2)
        while augmentations[0] == augmentations[1]:
            index = options_aug.index(augmentations[0])
            values.pop(index)
            options_aug.remove(augmentations[0])
            augmentations[1] = random.choices(options_aug, values, k=1)[0]
        aug1 = self.graph_augmentations[(graph_ind, augmentations[0])]
        aug2 = self.graph_augmentations[(graph_ind, augmentations[1])]
        aug1 = data_functions.add_self_loops(aug1)
        aug2 = data_functions.add_self_loops(aug2)
        return aug1, aug2


class ContrastiveAndClassificationModel(nn.Module):
    def __init__(
        self,
        input_size,
        hidden1,
        hidden2,
        embedding_size,
        hidden,
        n_classes,
        device=device,
    ):
        super(ContrastiveAndClassificationModel, self).__init__()
        self.embedding_size = embedding_size
        self.n_classes = n_classes
        self.device = device
        # Set the layers for each augmentation
        self.conv1 = GCNConv(in_channels=input_size, out_channels=hidden1)
        self.conv2 = GCNConv(in_channels=hidden1, out_channels=hidden2)
        self.fc0 = nn.Linear(in_features=hidden2, out_features=embedding_size)
        self.fc1 = nn.Linear(in_features=embedding_size, out_features=hidden)
        self.fc2 = nn.Linear(in_features=hidden, out_features=n_classes)

    def forward(self, graph):
        features, edge_index = graph
        features, edge_index = features.to(device).squeeze(0), edge_index.to(
            device
        ).squeeze(0)
        if len(features.shape) > 2:
            features = features.squeeze()
        if len(features.shape) == 1:
            features = features.unsqueeze(0)
        features = features.float()
        x_GCN = self.conv1(features, edge_index)
        x_GCN = F.relu(x_GCN)
        x_GCN = self.conv2(x_GCN, edge_index)
        x_GCN = F.relu(x_GCN)
        # pooling
        x_GCN = torch.max(x_GCN, dim=0)[0]
        # x_GCN = torch.mean(x_GCN, dim=0)
        x_embedding = self.fc0(x_GCN)
        # classification
        x = F.relu(self.fc1(x_embedding))
        x = self.fc2(x)
        return x_embedding, x

    def loss_graphcl(self, x1, x2):
        T = 0.5
        batch_size, _ = x1.size()

        x1_abs = x1.norm(dim=1)
        x2_abs = x2.norm(dim=1)

        sim_matrix = torch.einsum("ik,jk->ij", x1, x2) / torch.einsum(
            "i,j->ij", x1_abs, x2_abs
        )
        sim_matrix = torch.exp(sim_matrix / T)
        sim_matrix = sim_matrix.nan_to_num(0)
        pos_sim = sim_matrix[range(batch_size), range(batch_size)]
        loss = pos_sim / (sim_matrix.sum(dim=1) - pos_sim)
        loss = - torch.log(loss).mean()
        return loss


def train_and_test(G_train, X_augmentations_train, y_train, G_val, X_augmentations_val, y_val, model_params=params):
    n_classes = len(np.unique(list(y_train.values())))
    train_dataset = AugmentedGraphDataset(
        G_train, X_augmentations_train, y_train, p=model_params["percentage_dict"]
    )
    val_dataset = AugmentedGraphDataset(G_val, X_augmentations_val, y_val, p=model_params["percentage_dict"])
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)

    features_size = train_dataset.attributes_size()
    model = ContrastiveAndClassificationModel(
        input_size=features_size,
        hidden1=model_params["hidden1"],
        hidden2=model_params["hidden2"],
        embedding_size=model_params["embedding_size"],
        hidden=model_params["hidden"],
        n_classes=n_classes,
    )
    model = model.to(device)
    model = model.float()
    optimizer = optim.Adam(params=model.parameters(), lr=model_params["lr"])
    loss_train = []
    accuracy_train = []
    loss_valid = []
    accuracy_valid = []
    epoch = 0
    PRETRAIN = True
    t = time.time()
    while not is_converge(loss_valid, MAX_EPOCHS=model_params["MAX_EPOCHS"]) or PRETRAIN:
        if (is_converge(loss_valid,  MAX_EPOCHS=int(model_params["MAX_EPOCHS"]/2)) or time.time() - t > 60 * 3) and PRETRAIN:
            loss_valid = []
            PRETRAIN = False
            # print(f"Epoch {epoch} done pretrain")
        # loop over the dataset multiple times
        epoch += 1
        train_or_test_model(
            model,
            optimizer,
            model_params["batch_size"],
            train_loader,
            loss_train,
            accuracy_train,
            TrainOrTest="Train",
            pretrainMode=PRETRAIN
        )
        train_or_test_model(
            model,
            optimizer,
            model_params["batch_size"],
            val_loader,
            loss_valid,
            accuracy_valid,
            TrainOrTest="Test",
            pretrainMode=PRETRAIN
        )
        # print(
        #     f"Epoch: {epoch}, Loss Train: {loss_train[-1]:.3f}, Loss Valid {loss_train[-1]:.3f}"
        #     f" Accuracy Train {accuracy_train[-1]:.3f}, Accuracy Valid {accuracy_valid[-1]:.3f}"
        # )
        if time.time() - t > 60 * 6:
            break
    G_train1, train_labels = train_dataset.return_data()
    G_val1, val_labels = val_dataset.return_data()
    train_pred, train_pred_prob = predict(model, G_train1)
    val_pred, val_pred_prob = predict(model, G_val1)

    y_train = list(y_train.values())
    y_val = list(y_val.values())
    prediction = {"train": train_pred, "val": val_pred}
    accuracy_train = accuracy_score(y_train, train_pred)
    accuracy_val = accuracy_score(y_val, val_pred)


    f1_micro_train = f1_score(y_train, train_pred, average="micro")
    f1_micro_val = f1_score(y_val, val_pred, average="micro")
    f1_macro_train = f1_score(y_train, train_pred, average="macro")
    f1_macro_val = f1_score(y_val, val_pred, average="macro")
    recall_train = recall_score(y_train, train_pred, average="macro")
    recall_val = recall_score(y_train, train_pred, average="macro")
    # Auc
    if n_classes == 2:
        auc_train = roc_auc_score(y_train, train_pred_prob)
        auc_val = roc_auc_score(y_val, val_pred_prob)
    else:
        auc_train = None
        auc_val = None

    scores = {"Accuracy": {"train": accuracy_train, "val": accuracy_val},
              "F1_micro": {"train": f1_micro_train, "val": f1_micro_val},
              "F1_macro": {"train": f1_macro_train, "val": f1_macro_val},
              "AUC": {"train": auc_train, "val": auc_val},
              "Recall": {"train": recall_train, "val": recall_val}}

    return model, prediction, scores


def train_or_test_model(
    model, optimizer, batch_size, dataloader, loss_vec, accuracy_vec, TrainOrTest, pretrainMode
):
    if TrainOrTest == "Train":
        model.train()
        optimizer.zero_grad()
        run_model(
            model,
            optimizer,
            batch_size,
            dataloader,
            loss_vec,
            accuracy_vec,
            TrainOrTest,
            pretrainMode
        )
    else:
        model.eval()
        with torch.no_grad():
            run_model(
                model,
                optimizer,
                batch_size,
                dataloader,
                loss_vec,
                accuracy_vec,
                TrainOrTest,
                pretrainMode
            )

def run_model(
    model, optimizer, batch_size, dataloader, loss_vec, accuracy_vec, TrainOrTest, pretrainMode
):
    ce_loss_function = nn.CrossEntropyLoss()
    running_loss = 0.0
    running_accuracy = 0.0

    embeddings = torch.zeros(batch_size, model.embedding_size, device=device)
    predictions = torch.zeros(batch_size, model.n_classes, device=device)
    augmented_embeddings = torch.zeros(batch_size, model.embedding_size, device=device)
    augmented_predictions = torch.zeros(batch_size, model.n_classes, device=device)
    labels = torch.zeros(batch_size, model.n_classes, device=device)
    i = 0
    n_batches = 0
    for idx, (graph1, graph2, label) in enumerate(dataloader):
        graph1_embedding, graph1_prediction = model(graph1)
        embeddings[i, :] = graph1_embedding.unsqueeze(0)
        predictions[i, :] = graph1_prediction
        graph2_embedding, graph2_prediction = model(graph1)
        augmented_embeddings[i, :] = graph2_embedding.unsqueeze(0)
        augmented_predictions[i, :] = graph2_prediction
        labels[i, :] = label
        i += 1
        LAST_BATCH = (idx == dataloader.__len__()-1)
        if i == batch_size or LAST_BATCH and (i != 1):
            n_batches += 1
            if LAST_BATCH:
                embeddings = embeddings[:i, :]
                predictions = predictions[:i, :]
                augmented_embeddings = augmented_embeddings[:i, :]
                augmented_predictions = augmented_predictions[:i, :]
                labels = labels[:i, :]
            if pretrainMode:
                loss = model.loss_graphcl(embeddings, augmented_embeddings)
            else:
                loss = (
                               ce_loss_function(predictions.float(), labels)
                )
                       #         +
                       #  ce_loss_function(augmented_predictions.float(), labels)
                       # ) / 2


            if TrainOrTest == "Train":
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

            running_loss += loss.item()
            predictions = nn.Softmax()(predictions)
            running_accuracy += (accuracy_score(
                labels.detach().cpu().numpy().argmax(1),
                predictions.detach().cpu().numpy().argmax(1))
            )
            # ) + accuracy_score(
            #     labels.detach().cpu().numpy().argmax(1),
            #     augmented_predictions.detach().cpu().numpy().argmax(1),
            # ))/2
            embeddings = torch.zeros(batch_size, model.embedding_size, device=device)
            predictions = torch.zeros(batch_size, model.n_classes, device=device)
            augmented_embeddings = torch.zeros(batch_size, model.embedding_size, device=device)
            augmented_predictions = torch.zeros(batch_size, model.n_classes, device=device)
            labels = torch.zeros(batch_size, model.n_classes, device=device)
            i = 0
    loss_vec.append(running_loss / n_batches)
    accuracy_vec.append(running_accuracy / n_batches)


def predict(model, X):
    predictions = []
    pred_scores = []
    for graph in X:
        embedding, pred = model(graph)
        pred_prob = nn.Softmax()(pred)
        pred = pred_prob.detach().cpu().numpy()
        predictions.append(pred.argmax())
        pred_scores.append(pred[1])
    return np.array(predictions), np.array(pred_scores)


def is_converge(vector, MAX_EPOCHS=20):
    if len(vector) > MAX_EPOCHS:
        return True
    if len(vector) < 10:
        return False
    if min(vector) < min(vector[-10:]):
        return True
    else:
        return False

