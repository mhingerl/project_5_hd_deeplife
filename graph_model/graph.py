import json
from pathlib import Path
from tqdm import tqdm
import numpy as np
import os
import torch
from torch_geometric.data import Dataset as GraphDataset
from torch_geometric.loader import DataLoader as GraphDataLoader
from torch_geometric.data import Data as GraphData
from calculate_pairwise_distances import load_compressed_tensor_dict
from sklearn import metrics

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.nn import Sequential as GraphSequential
from torch_geometric.nn.norm import LayerNorm, BatchNorm


class PickledGraphDataset(GraphDataset):
    def __init__(self, root, transform=None, pre_transform=None):
        super(PickledGraphDataset, self).__init__(root, transform, pre_transform)
        self.file_list = sorted([f for f in os.listdir(root) if f.endswith(".gz")])

    def len(self):
        return len(self.file_list)

    def get(self, idx):
        file_path = os.path.join(self.root, self.file_list[idx])
        data_dict = load_compressed_tensor_dict(file_path)
        data = GraphData.from_dict(data_dict)
        return data


def GCNBlock(in_channels, out_channels):
    return GraphSequential(
        "x, edge_index",
        [
            (nn.Dropout(p=0.6), "x -> x"),
            (BatchNorm(in_channels), "x -> x"),
            (GCNConv(in_channels, out_channels), "x, edge_index -> x"),
            (nn.ReLU(), "x -> x"),
        ],
    )


class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, n_hidden_layers, num_classes):
        super().__init__()
        self.input_conv = GCNBlock(in_channels, hidden_channels)
        self.hidden_convs = nn.ModuleList(
            [GCNBlock(hidden_channels, hidden_channels) for _ in range(n_hidden_layers)]
        )
        self.out_layer = nn.Linear(hidden_channels, num_classes)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = self.input_conv(x, edge_index)
        for hidden_conv in self.hidden_convs:
            x = hidden_conv(x, edge_index)
        logits = self.out_layer(x)
        return logits.squeeze()


from sklearn.utils import class_weight


def compute_class_weights(labels):
    class_weights = class_weight.compute_class_weight(
        class_weight="balanced", classes=np.unique(labels), y=labels
    )
    class_weights = torch.tensor(class_weights, dtype=torch.float)
    return class_weights


def get_roc_auc(y, logits):
    fpr, tpr, thresholds = metrics.roc_curve(
        y.cpu().numpy(), torch.sigmoid(logits).cpu().numpy()
    )
    roc_auc = metrics.auc(fpr, tpr)
    return roc_auc


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_dataset = PickledGraphDataset(
        root=f"/workspace/graphs/holo/sc/full_train_data"
    )
    train_dataloader = GraphDataLoader(train_dataset, batch_size=64, shuffle=True)

    test_dataset = PickledGraphDataset(root="/workspace/graphs/holo/sc/test")
    test_dataloader = GraphDataLoader(test_dataset, batch_size=64, shuffle=False)

    test_apo_dataset = PickledGraphDataset(root="/workspace/graphs/apo/sc/test")
    test_apo_dataloader = GraphDataLoader(
        test_apo_dataset, batch_size=64, shuffle=False
    )

    all_ys = torch.tensor(np.load("/workspace/train_ys_fold0.npy"))
    class_weights = compute_class_weights(all_ys.numpy()).to(device)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=class_weights[1])

    model = GCN(
        in_channels=1536,
        hidden_channels=256,
        n_hidden_layers=4,
        num_classes=1,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    holo_test_loss_array = []
    holo_roc_auc_array = []
    apo_test_loss_array = []
    apo_roc_auc_array = []

    model.train()
    full_train_loss = []
    for epoch in range(20):
        epoch_losses = []
        pbar = tqdm(enumerate(train_dataloader), total=len(train_dataloader))
        for i, data in pbar:
            data = data.to(device)
            optimizer.zero_grad()
            logits = model(data)
            loss = loss_fn(logits, data.y.float())
            epoch_losses.append(loss.item())
            loss.backward()
            full_train_loss.append(loss.item())
            optimizer.step()
            roc_auc = get_roc_auc(data.y.detach().float(), logits.detach())

            if i % 2 == 0:
                pbar.set_description(f"Loss: {loss.item()}, ROC AUC: {roc_auc}")
                pbar.refresh()

        if epoch % 1 == 0:
            for name, test_datal in {
                "holo": test_dataloader,
                "apo": test_apo_dataloader,
            }.items():
                model.eval()
                test_loss = 0.0
                roc_auc = 0.0

                with torch.no_grad():
                    for data in tqdm(
                        test_datal, total=len(test_datal), desc=f"Evaluation {name}"
                    ):
                        data = data.to(device)
                        logits = model(data)
                        test_loss += loss_fn(logits, data.y.float()).item()
                        roc_auc += get_roc_auc(data.y.float(), logits)

                print(f"Test Loss {name}: {test_loss / len(test_datal)}")
                print(f"ROC AUC {name}: {roc_auc / len(test_datal)}")

                if name == "holo":
                    holo_test_loss_array.append(test_loss / len(test_datal))
                    holo_roc_auc_array.append(roc_auc / len(test_datal))
                if name == "apo":
                    apo_test_loss_array.append(test_loss / len(test_datal))
                    apo_roc_auc_array.append(roc_auc / len(test_datal))

            OUT = Path(f"/workspace/evaluation/full_train_data")
            OUT.mkdir(exist_ok=True, parents=True)

            np.save(OUT / f"full_train_loss.npy", full_train_loss)
            np.save(
                OUT / "holo_test_loss_array.npy",
                np.array(holo_test_loss_array),
            )
            np.save(
                OUT / "holo_roc_auc_array.npy",
                np.array(holo_roc_auc_array),
            )
            np.save(
                OUT / "apo_test_loss_array.npy",
                np.array(apo_test_loss_array),
            )
            np.save(
                OUT / "apo_roc_auc_array.npy",
                np.array(apo_roc_auc_array),
            )
