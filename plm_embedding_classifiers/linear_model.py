import json
from loguru import logger
from tqdm import tqdm
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torch import nn
from sklearn.utils import class_weight
from sklearn import metrics

from residues import process_sequence_dataset


class SequenceDataset(Dataset):
    def __init__(self, _Xs, _Ys):
        if _Xs is None and _Ys is None:
            logger.info("Creating Empty Sequence Dataset...")
            return
        logger.info("Creating Sequence Dataset...")
        Xs_list = None
        Ys_list = None
        for key in tqdm(_Xs.keys()):
            if Xs_list is None:
                Xs_list = _Xs[key]
                Ys_list = _Ys[key]
            else:
                Xs_list = np.concatenate((Xs_list, _Xs[key]), axis=0)
                Ys_list = np.concatenate((Ys_list, _Ys[key]), axis=0)

        self.Xs = torch.tensor(Xs_list, dtype=torch.float32)
        self.Ys = torch.tensor(Ys_list, dtype=torch.int64)

    def __len__(self):
        assert len(self.Xs) == len(self.Ys)
        return len(self.Xs)

    def __getitem__(self, idx):
        x = self.Xs[idx]
        y = self.Ys[idx]
        return x, y


def lazy_lin_block(out_features, dropout):
    return nn.Sequential(
        nn.LazyLinear(out_features=out_features),
        nn.ReLU(),
        nn.Dropout(dropout),
    )


class Embed2BindingSiteClassifier(nn.Module):
    def __init__(self, layers, dropout, out_features):
        super().__init__()
        self.layers = nn.ModuleList(
            [lazy_lin_block(out_dim, dropout) for out_dim in layers]
        )
        self.out_projection = nn.LazyLinear(out_features=out_features)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        out = self.out_projection(x)
        return out.squeeze()


def compute_class_weights(labels):
    class_weights = class_weight.compute_class_weight(
        class_weight="balanced", classes=np.unique(labels), y=labels
    )
    class_weights = torch.tensor(class_weights, dtype=torch.float)
    return class_weights


def train(model, optimizer, loss_fn, train_loader):
    """Train model for one epoch."""
    model.train()

    losses_epoch = []
    for X_train, y_train in train_loader:
        X_train, y_train = X_train.to(device), y_train.to(device)
        y_train = y_train.float()
        y_logits = model(X_train)
        loss = loss_fn(y_logits, y_train)
        losses_epoch.append(loss.item())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return model, optimizer, np.array(losses_epoch).mean()


def eval(model, loss_fn, test_loader):
    """Evaluate model on test set"""
    model.eval()
    test_losses = []

    for X_test, y_test in test_loader:
        with torch.no_grad():
            X_test, y_test = X_test.to(device), y_test.to(device)
            y_test = y_test.float()
            y_logits = model(X_test)
            test_loss = loss_fn(y_logits, y_test)
            test_losses.append(test_loss.item())

    return np.array(test_losses).mean()


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(42)

    model = Embed2BindingSiteClassifier(
        layers=[100, 100],
        dropout=0.4,
        out_features=1,
    ).to(device)

    epochs = 20

    # Create an optimizer
    optimizer = torch.optim.Adam(params=model.parameters(), lr=0.001)

    # with open("/workspace/CryptoBench/single_chain/train/train-fold-0.json") as f:
    #     split_annotations = json.load(f)

    # with open("/workspace/pdb_bullshit/extracted_sequences/holo_sc_seqs.json") as f:
    #     extracted_info = json.load(f)

    # X_train, Y_train = process_sequence_dataset(
    #     split_annotations=split_annotations,
    #     extracted_info=extracted_info,
    #     embeddings_path=Path("/workspace/uniprot_embeddings/ankh/sc"),
    #     fasta_path=Path("/workspace/fastas/sc"),
    # )

    # train_dataset = SequenceDataset(X_train, Y_train)

    # with open("/workspace/CryptoBench/single_chain/test/test.json") as f:
    #     split_annotations = json.load(f)

    # with open("/workspace/pdb_bullshit/extracted_sequences/holo_sc_seqs.json") as f:
    #     extracted_info = json.load(f)

    # X_test, Y_test = process_sequence_dataset(
    #     split_annotations=split_annotations,
    #     extracted_info=extracted_info,
    #     embeddings_path=Path("/workspace/uniprot_embeddings/ankh/sc"),
    #     fasta_path=Path("/workspace/fastas/sc"),
    # )

    # test_dataset = SequenceDataset(X_test, Y_test)
    # test_xs, test_ys = test_dataset[:]
    # np.save("test_xs.npy", test_xs.numpy())
    # np.save("test_ys.npy", test_ys.numpy())

    test_dataset = SequenceDataset(None, None)
    test_dataset.Xs = torch.tensor(np.load("test_xs.npy"))
    test_dataset.Ys = torch.tensor(np.load("test_ys.npy"))
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

    train_dataset = SequenceDataset(None, None)
    train_dataset.Xs = torch.tensor(np.load("train_xs_fold0.npy"))
    train_dataset.Ys = torch.tensor(np.load("train_ys_fold0.npy"))
    train_loader = DataLoader(train_dataset, batch_size=512, shuffle=True)

    # compute class weights (because the dataset is heavily imbalanced)

    train_xs, train_ys = train_dataset[:]
    # np.save("train_xs_fold0.npy", train_xs.numpy())
    # np.save("train_ys_fold0.npy", train_ys.numpy())

    _, train_ys = train_dataset[:]
    class_weights = compute_class_weights(train_ys.numpy()).to(device)

    # BCEWithLogitsLoss - sigmoid is already built-in!
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=class_weights[1])
    train_losses = []
    test_losses = []

    train_loss = 0
    test_loss = 0
    pbar = tqdm(range(epochs))
    for epoch in pbar:
        pbar.set_description(
            f"Epoch: {epoch}, Train Loss: {train_loss:.5f}, Test Loss: {test_loss}"
        )
        pbar.refresh()
        model, optimizer, train_loss = train(model, optimizer, loss_fn, train_loader)
        test_loss = eval(model, loss_fn, test_loader)

        train_losses.append(train_loss)
        test_losses.append(test_loss)
