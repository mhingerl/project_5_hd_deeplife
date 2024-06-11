"""
Utility file to embed protein data using ankh.

Run as script by providing a file with protein sequences separated by \\n.

Example:
    python ankh.py --input_sequences data.json --output_dir embeddings --type numpy
"""

import argparse

from loguru import logger
import numpy as np
import torch

import ankh
from protein_lms.utils import save_tensor


def embed_protein_ankh(protein_sequences: list[str]) -> torch.Tensor:
    """
    Embeds protein sequences using ankh model.

    Args:
        protein_sequence: list of protein sequences in single letter notation.

    Returns:
        embeddings: torch.Tensor of embeddings.
    """
    logger.info("Loading ankh model.")
    model, tokenizer = ankh.load_large_model()
    model.eval()

    protein_sequences = [list(seq) for seq in protein_sequences]

    logger.info("Tokenizing protein sequences.")
    outputs = tokenizer.batch_encode_plus(
        protein_sequences,
        add_special_tokens=True,
        padding=True,
        is_split_into_words=True,
        return_tensors="pt",
    )

    logger.info("Embedding protein sequences.")
    with torch.no_grad():
        embeddings = model(
            input_ids=outputs["input_ids"], attention_mask=outputs["attention_mask"]
        )

    return embeddings


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("input_sequences", type=str, help="Path to input file.")
    parser.add_argument("output_dir", type=str, help="Path to output directory.")
    parser.add_argument(
        "--type",
        type=str,
        default="numpy",
        choices=["numpy", "torch"],
        help="Type of output file.",
    )

    args = parser.parse_args()

    logger.info(f"Reading protein sequences from {args.input_sequences}.")
    with open(args.input_sequences, "r") as f:
        protein_sequences = f.read().splitlines()

    embeddings = embed_protein_ankh(protein_sequences)
    save_tensor(embeddings=embeddings, output_dir=args.output_dir, type=args.type)
