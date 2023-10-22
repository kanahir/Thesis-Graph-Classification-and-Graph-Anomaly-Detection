import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, recall_score
import time
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model_params = {"batch_size": 32,
                "lr": 0.01,
                "MAX_EPOCHS": 35,
                "hidden": 40
                }

class EmbeddingDataset(Dataset):
    def __init__(self, matrix, labels, device=device):
        self.device = device
        self.matrix = matrix.to(device)
        self.labels = labels.to(device)

    def __len__(self):
        return self.matrix.shape[0]

    def __getitem__(self, idx):
        vector = self.matrix[idx, :]
        label = self.labels[idx]
        return vector, label


class GraphClassificationModel(nn.Module):
    def __init__(self, input_size, hidden, n_classes, device=device):
        super(GraphClassificationModel, self).__init__()
        self.device = device
        # Set the layers for each augmentation
        self.fc1 = nn.Linear(in_features=input_size, out_features=hidden)
        self.fc2 = nn.Linear(in_features=hidden, out_features=n_classes)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


def train_and_test(train_embedding, y_train, validation_embedding, y_val, model_params=model_params):
    n_classes = len(np.unique(y_train))
    train_embedding = torch.tensor(train_embedding, device=device)
    validation_embedding = torch.tensor(validation_embedding, device=device)
    y_train = torch.tensor(np.array(y_train), device=device)
    y_val = torch.tensor(np.array(y_val), device=device)
    # normalization
    mu = torch.mean(train_embedding, dim=0)
    std = torch.std(train_embedding, dim=0)
    x_train_normalized = (train_embedding - mu) / std
    x_valid_normalized = (validation_embedding - mu) / std

    train_dataset = EmbeddingDataset(x_train_normalized, y_train)
    val_dataset = EmbeddingDataset(x_valid_normalized, y_val)
    train_loader = DataLoader(train_dataset, batch_size=model_params["batch_size"], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=model_params["batch_size"], shuffle=False)

    input_size = train_embedding.shape[1]
    model = GraphClassificationModel(input_size=input_size, hidden=model_params["hidden"],
                                     n_classes=n_classes)
    model = model.to(device)
    model = model.float()
    optimizer = optim.Adam(params=model.parameters(), lr=model_params["lr"])
    loss_train = []
    loss_valid = []
    accuracy_train = []
    accuracy_valid = []
    epoch = 0
    t = time.time()
    while epoch <= model_params["MAX_EPOCHS"] and not is_converge(loss_valid, MAX_EPOCHS=model_params["MAX_EPOCHS"]):
        # loop over the dataset multiple times
        epoch += 1
        train_model(model, optimizer, train_loader, loss_train, accuracy_train)
        test_model(model, val_loader, loss_valid, accuracy_valid)

        print(
            "Epoch: {}, Loss Train: {:.3f}, Loss Valid {:.3f}, Scores Train: {:.3f}, Scores Valid {:.3f}".format(
                epoch, loss_train[-1], loss_valid[-1], accuracy_train[-1], accuracy_valid[-1]
            )
        )
        if time.time() - t > 60 * 1.5:
            break
    train_pred = model(x_train_normalized).detach().cpu().numpy().argmax(1)
    val_pred = model(x_valid_normalized).detach().cpu().numpy().argmax(1)
    prediction = {"train": train_pred, "val": val_pred}
    y_train = y_train.detach().cpu().numpy()
    y_val = y_val.detach().cpu().numpy()
    accuracy_train = accuracy_score(y_train, train_pred)
    accuracy_val = accuracy_score(y_val, val_pred)
    f1_micro_train = f1_score(y_train, train_pred, average="micro")
    f1_micro_val = f1_score(y_val, val_pred, average="micro")
    f1_macro_train = f1_score(y_train, train_pred, average="macro")
    f1_macro_val = f1_score(y_val, val_pred, average="macro")
    recall_train = recall_score(y_train, train_pred, average="weighted")
    recall_val = recall_score(y_train, train_pred, average="weighted")
    # Auc
    if n_classes == 2:
        train_pred_prob = model(x_train_normalized).detach().cpu().numpy()[:, 1]
        val_pred_prob = model(x_valid_normalized).detach().cpu().numpy()[:, 1]
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


def train_model(model, optimizer, dataloader, loss_train, accuracy_train):
    loss_func = nn.CrossEntropyLoss()
    running_loss = 0.0
    running_accuracy = 0.0
    optimizer.zero_grad()
    for idx, batch in enumerate(dataloader):
        optimizer.zero_grad()
        data, labels = batch
        pred = model(data)
        loss = loss_func(pred.float(), labels.long())
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        running_accuracy += accuracy_score(labels.detach().cpu().numpy(), pred.detach().cpu().numpy().argmax(1))
    loss_train.append(running_loss / (idx + 1))
    accuracy_train.append(running_accuracy / (idx + 1))


def test_model(model, dataloader, loss_valid, accuracy_valid):
    loss_func = nn.CrossEntropyLoss()
    running_loss = 0.0
    running_accuracy = 0.0
    i = 0
    for batch_idx, batch in enumerate(dataloader):
        data, labels = batch
        i += 1
        pred = model(data)
        loss = loss_func(pred.float(), labels.long())
        running_loss += loss.item()
        running_accuracy += accuracy_score(labels.detach().cpu().numpy(), pred.detach().cpu().numpy().argmax(1))
    loss_valid.append(running_loss / (batch_idx + 1))
    accuracy_valid.append(running_accuracy / (batch_idx + 1))

def is_converge(vector, MAX_EPOCHS=20):
    if len(vector) > MAX_EPOCHS:
        return True
    if len(vector) < 10:
        return False
    if min(vector) < min(vector[-10:]):
        return True
    else:
        return False
