import random
import numpy as np
import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import time

import data.data_functions as data_functions
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class AugmentedGraphDataset(Dataset):
    def __init__(self, graphs, graph_augmentations, p, device=device):
        graphs_index = list(graphs.keys())
        self.get_index_dict = {new_index: graph_index for new_index, graph_index in enumerate(graphs_index)}
        self.device = device
        self.probability_dict = p
        # add self loops
        self.graphs = {i: data_functions.add_self_loops(g) for i, g in graphs.items()}
        self.graph_augmentations = graph_augmentations

    def __len__(self):
        return len(self.graphs)

    def __getitem__(self, idx):
        graph_ind = self.get_index_dict[idx]
        graph = self.graphs[graph_ind]
        aug1, aug2 = self.get_augmentations(graph_ind, self.probability_dict)
        attr1 = data_functions.get_attributes(aug1).to(self.device)
        edges1 = data_functions.get_edges(aug1)
        attr2 = data_functions.get_attributes(aug2).to(self.device)
        edges2 = data_functions.get_edges(aug2)
        return (attr1, edges1), (attr2, edges2)

    def return_data(self):
        graphs = []
        for i in range(len(self)):
            graph_ind = self.get_index_dict[i]
            graph = self.graph_augmentations[(graph_ind, "ID")]
            graph = data_functions.add_self_loops(graph)
            attr = data_functions.get_attributes(graph).to(self.device)
            edges = data_functions.get_edges(graph)
            graphs.append((attr, edges))
        return graphs

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


class ContrastiveLearningGraphModel(nn.Module):
    def __init__(self, input_size, hidden1, hidden2, embedding_size, device=device):
        super(ContrastiveLearningGraphModel, self).__init__()
        self.embedding_size = embedding_size
        self.device = device
        # Set the layers for each augmentation
        self.conv1 = GCNConv(in_channels=input_size, out_channels=hidden1)
        self.conv2 = GCNConv(in_channels=hidden1, out_channels=hidden2)
        self.fc = nn.Linear(in_features=hidden2, out_features=embedding_size)
        self.loss_function = self.loss_graphcl

    def forward(self, graph):
        features, edge_index = graph
        features, edge_index = features.to(device).squeeze(0), edge_index.to(device).squeeze(0)
        if len(features.shape) > 2:
            features = features.squeeze()
        if len(features.shape) == 1:
            features = features.unsqueeze(0)
        features = features.float()
        x_GCN = self.conv1(features, edge_index)
        x_GCN = F.relu(x_GCN)
        x_GCN = self.conv2(x_GCN, edge_index)
        x_GCN = torch.max(x_GCN, dim=0)[0]
        x_GCN = self.fc(x_GCN)
        return x_GCN

    def loss_graphcl(self, x1, x2):
        T = 0.5
        batch_size, _ = x1.size()

        x1_abs = x1.norm(dim=1)
        x2_abs = x2.norm(dim=1)

        sim_matrix = torch.einsum('ik,jk->ij', x1, x2) / torch.einsum('i,j->ij', x1_abs, x2_abs)
        sim_matrix = torch.exp(sim_matrix / T)
        pos_sim = sim_matrix[range(batch_size), range(batch_size)]
        loss = pos_sim / (sim_matrix.sum(dim=1) - pos_sim)
        loss = - torch.log(loss).sum()
        return loss

    def predict(self, graphs_list):
        n_graphs = len(graphs_list)
        embeddings = np.zeros([n_graphs, self.embedding_size])
        for i, graph in enumerate(graphs_list):
            embedding = self.forward(graph).detach().cpu().numpy()
            embeddings[i, :] = embedding
        return embeddings


def train_and_test(G_train, X_augmentations_train, G_val, X_augmentations_val, model_params):
    train_dataset = AugmentedGraphDataset(G_train, X_augmentations_train, p=model_params["percentage_dict"])
    val_dataset = AugmentedGraphDataset(G_val, X_augmentations_val, p=model_params["percentage_dict"])
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)

    features_size = train_dataset.attributes_size()
    model = ContrastiveLearningGraphModel(input_size=features_size, hidden1=model_params["hidden1"],
                                          hidden2=model_params["hidden2"], embedding_size=model_params["embedding_size"])
    model = model.to(device)
    model = model.float()
    optimizer = optim.Adam(params=model.parameters(), lr=model_params["lr"])
    loss_train = []
    loss_valid = []
    epoch = 0
    t = time.time()
    while not is_converge(loss_valid, model_params["MAX_EPOCHS"]):
        # loop over the dataset multiple times
        epoch += 1
        train_model(model, optimizer, model_params["batch_size"], train_loader, loss_train)
        test_model(model,  model_params["batch_size"], val_loader, loss_valid)
        print(
            "Epoch: {}, Loss Train: {:.3f}, Loss Valid {:.3f}".format(
                epoch, loss_train[-1], loss_valid[-1]
            )
        )
        if time.time() - t > 60 * 5:
            break
    train = DataLoader(train_dataset, batch_size=1, shuffle=False)
    val = DataLoader(val_dataset, batch_size=1, shuffle=False)
    train_embedding = predict(model, train)
    validation_embedding = predict(model, val)
    return model, train_embedding, validation_embedding

def train_model(model, optimizer, batch_size, dataloader, loss_train):
    running_loss = 0.0
    embeddings = torch.zeros(batch_size, model.embedding_size)
    augmented_embeddings = torch.zeros(batch_size, model.embedding_size)
    i = 0
    optimizer.zero_grad()
    for idx, (graph1, graph2) in enumerate(dataloader):
        embeddings[i, :] = model(graph1).unsqueeze(0)
        augmented_embeddings[i, :] = model(graph2).unsqueeze(0)
        i += 1
        LAST_BATCH = (idx == dataloader.__len__()-1)
        if i == batch_size or LAST_BATCH:
            if LAST_BATCH:
                embeddings = embeddings[:i, :]
                augmented_embeddings = augmented_embeddings[:i, :]
            loss = model.loss_function(embeddings, augmented_embeddings)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            running_loss += loss.item()
            embeddings = torch.zeros(batch_size, model.embedding_size)
            augmented_embeddings = torch.zeros(batch_size, model.embedding_size)
            i = 0
    #     last batch
    loss_train.append(running_loss / len(dataloader.dataset))


def test_model(model, batch_size, dataloader, loss_valid):
    running_loss = 0.0
    embeddings = torch.zeros(batch_size, model.embedding_size)
    augmented_embeddings = torch.zeros(batch_size, model.embedding_size)
    i = 0
    for batch_idx, (graph, augmented_graph) in enumerate(dataloader):
        embeddings[i, :] = model(graph).unsqueeze(0)
        augmented_embeddings[i, :] = model(augmented_graph).unsqueeze(0)
        i += 1
        LAST_BATCH = (batch_idx == dataloader.__len__() - 1)
        if i == batch_size or LAST_BATCH and (i != 1):
            if LAST_BATCH:
                embeddings = embeddings[:i, :]
                augmented_embeddings = augmented_embeddings[:i, :]
            loss = model.loss_function(embeddings, augmented_embeddings)
            running_loss += loss.item()
            embeddings = torch.zeros(batch_size, model.embedding_size)
            augmented_embeddings = torch.zeros(batch_size, model.embedding_size)
            i = 0
    loss_valid.append(running_loss / len(dataloader.dataset))


def predict(model, dataloader):
    predictions = []
    for graph, augmented_graph in dataloader:
        embedding = model(graph).detach().cpu().numpy()
        predictions.append(embedding)
    return np.array(predictions)


def is_converge(vector, MAX_EPOCHS=20):
    if len(vector) > MAX_EPOCHS:
        return True
    if len(vector) < 10:
        return False
    if min(vector) < min(vector[-10:]):
        return True
    else:
        return False
