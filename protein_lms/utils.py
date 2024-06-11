from loguru import logger
import numpy as np
import torch


def save_tensor(embeddings: torch.Tensor, output_dir: str, type: str) -> None:
    if type == "numpy":
        embeddings = embeddings.cpu().numpy()
        logger.info(f"Saving embeddings to {output_dir}.")
        np.save(output_dir, embeddings)
    elif type == "torch":
        logger.info(f"Saving embeddings to {output_dir}.")
        torch.save(embeddings, output_dir)
