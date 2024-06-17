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
    auc_epoch = []
    for X_train, y_train in train_loader:
        X_train, y_train = X_train.to(device), y_train.to(device)
        y_train = y_train.float()
        y_logits = model(X_train)
        loss = loss_fn(y_logits, y_train)
        losses_epoch.append(loss.item())
        auc_epoch.append(get_roc_auc(y_train.detach(), y_logits.detach()))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return model, optimizer, np.array(losses_epoch).mean(), np.array(auc_epoch).mean()


def eval(model, loss_fn, test_loader):
    """Evaluate model on test set"""
    model.eval()
    test_losses = []
    auc_epoch = []

    for X_test, y_test in test_loader:
        with torch.no_grad():
            X_test, y_test = X_test.to(device), y_test.to(device)
            y_test = y_test.float()
            y_logits = model(X_test)
            test_loss = loss_fn(y_logits, y_test)
            test_losses.append(test_loss.item())
            auc_epoch.append(get_roc_auc(y_test, y_logits))

    return np.array(test_losses).mean(), np.array(auc_epoch).mean()


def save_sequence_dataset(
    dataset_json, extracted_sequences_json, model, mode, save_path, file_prefix
):
    with open(f"/workspace/CryptoBench/single_chain/{dataset_json}") as f:
        split_annotations = json.load(f)

    with open(
        f"/workspace/pdb_bullshit/extracted_sequences/{extracted_sequences_json}"
    ) as f:
        extracted_info = json.load(f)

    X_test, Y_test = process_sequence_dataset(
        split_annotations=split_annotations,
        extracted_info=extracted_info,
        embeddings_path=Path(f"/workspace/uniprot_embeddings/{model}/sc"),
        fasta_path=Path("/workspace/fastas/sc"),
        mode=mode,
    )

    test_dataset = SequenceDataset(X_test, Y_test)
    test_xs, test_ys = test_dataset[:]
    save_path.mkdir(exist_ok=True, parents=True)
    np.save(save_path / f"{file_prefix}_xs.npy", test_xs.numpy())
    np.save(save_path / f"{file_prefix}_ys.npy", test_ys.numpy())


def get_roc_auc(y, logits):
    fpr, tpr, thresholds = metrics.roc_curve(
        y.cpu().numpy(), torch.sigmoid(logits).cpu().numpy()
    )
    roc_auc = metrics.auc(fpr, tpr)
    return roc_auc


if __name__ == "__main__":
    from itertools import product

    models = ["ankh", "t5"]
    folds = ["0", "1", "2", "3", "4"]

    for model_name, fold in product(models, folds):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        torch.manual_seed(42)

        model = Embed2BindingSiteClassifier(
            layers=[100, 100],
            dropout=0.4,
            out_features=1,
        ).to(device)

        epochs = 10

        # Create an optimizer
        optimizer = torch.optim.Adam(params=model.parameters(), lr=0.001)

        MODEL_DIR = Path("/workspace/sequence_datasets") / model_name
        train_dataset = SequenceDataset(None, None)
        train_dataset.Xs = torch.tensor(
            np.load(MODEL_DIR / "holo" / f"train-fold-{fold}_xs.npy")
        )
        train_dataset.Ys = torch.tensor(
            np.load(MODEL_DIR / "holo" / f"train-fold-{fold}_ys.npy")
        )
        train_loader = DataLoader(train_dataset, batch_size=1048, shuffle=True)

        test_dataset_holo = SequenceDataset(None, None)
        test_dataset_holo.Xs = torch.tensor(np.load(MODEL_DIR / "holo" / "test_xs.npy"))
        test_dataset_holo.Ys = torch.tensor(np.load(MODEL_DIR / "holo" / "test_ys.npy"))
        test_loader_holo = DataLoader(test_dataset_holo, batch_size=1048, shuffle=False)

        test_dataset_apo = SequenceDataset(None, None)
        test_dataset_apo.Xs = torch.tensor(np.load(MODEL_DIR / "apo" / "test_xs.npy"))
        test_dataset_apo.Ys = torch.tensor(np.load(MODEL_DIR / "apo" / "test_ys.npy"))
        test_loader_apo = DataLoader(test_dataset_apo, batch_size=1048, shuffle=False)

        # compute class weights (because the dataset is heavily imbalanced)
        _, train_ys = train_dataset[:]
        class_weights = compute_class_weights(train_ys.numpy()).to(device)

        # BCEWithLogitsLoss - sigmoid is already built-in!
        loss_fn = nn.BCEWithLogitsLoss(pos_weight=class_weights[1])
        train_losses_arr = []
        train_auc_arr = []
        test_losses_holo_arr = []
        test_auc_holo_arr = []
        test_losses_apo_arr = []
        test_auc_apo_arr = []

        train_loss = 0
        test_loss_holo = 0
        test_loss_apo = 0
        train_auc = 0
        test_auc_holo = 0
        test_auc_apo = 0
        pbar = tqdm(range(epochs))
        for epoch in pbar:
            pbar.set_description(
                f"Epoch: {epoch}, Train Loss: {train_loss:.5f}, Test Loss: {test_loss_holo:.5f}"
                f"Train AUC: {train_auc:.5f}, Test AUC: {test_auc_holo:.5f}"
            )
            pbar.refresh()
            model, optimizer, train_loss, train_auc = train(
                model, optimizer, loss_fn, train_loader
            )
            test_loss_holo, test_auc_holo = eval(model, loss_fn, test_loader_holo)
            test_loss_apo, test_auc_apo = eval(model, loss_fn, test_loader_apo)

            train_losses_arr.append(train_loss)
            train_auc_arr.append(train_auc)
            test_losses_holo_arr.append(test_loss_holo)
            test_auc_holo_arr.append(test_auc_holo)
            test_losses_apo_arr.append(test_loss_apo)
            test_auc_apo_arr.append(test_auc_apo)

            OUTDIR = Path("/workspace/plm_evals") / model_name
            OUTDIR.mkdir(exist_ok=True, parents=True)
            np.save(OUTDIR / f"train_losses_fold{fold}.npy", train_losses_arr)
            np.save(OUTDIR / f"train_auc_fold{fold}.npy", train_auc_arr)
            np.save(OUTDIR / f"test_losses_holo_fold{fold}.npy", test_losses_holo_arr)
            np.save(OUTDIR / f"test_auc_holo_fold{fold}.npy", test_auc_holo_arr)
            np.save(OUTDIR / f"test_losses_apo_fold{fold}.npy", test_losses_apo_arr)
            np.save(OUTDIR / f"test_auc_apo_fold{fold}.npy", test_auc_apo_arr)
