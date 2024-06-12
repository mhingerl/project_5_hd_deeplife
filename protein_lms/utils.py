from loguru import logger
import numpy as np
import torch
from torch.utils.data import Dataset


class ProteinDataset(Dataset):
    def __init__(self, sequence_dict) -> None:
        super().__init__()
        self.sequences = list(sequence_dict.values())
        self.filenames = list(sequence_dict.keys())

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return self.filenames[idx], self.sequences[idx]


def save_tensor(embeddings: torch.Tensor, output_dir: str, type: str) -> None:
    if type == "numpy":
        embeddings = embeddings.cpu().numpy()
        logger.info(f"Saving embeddings to {output_dir}.")
        np.save(output_dir, embeddings)
    elif type == "torch":
        logger.info(f"Saving embeddings to {output_dir}.")
        torch.save(embeddings, output_dir)


def subset_embeddings_with_attention_mask(
    embeddings: torch.Tensor, attention_mask: torch.Tensor
) -> np.ndarray:
    """
    Subset embeddings using attention mask.

    Args:
        embeddings: torch.Tensor of embeddings.
        attention_mask: torch.Tensor of attention mask.

    Returns:
        embeddings: torch.Tensor of embeddings.
    """
    for i in range(attention_mask.shape[0]):
        sequence_w_eos = embeddings[i, attention_mask[i] == 1]
        return sequence_w_eos[:-1].cpu().numpy()
