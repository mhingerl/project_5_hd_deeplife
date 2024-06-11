"""
Embedding protein sequences using the t5 model.

Args:
    protein_sequence: list of protein sequences in single letter notation.
    output_dir: Path to output directory.
    type: Type of output file.

Example:
    python prot_t5.py --input_sequences data.json --output_dir embeddings --type numpy
"""

import argparse
import re

from loguru import logger
import numpy as np
import torch
from transformers import T5Tokenizer, T5EncoderModel

from protein_lms.utils import save_tensor


def embed_protein_t5_xl(protein_sequences: list[str]) -> torch.Tensor:
    """
    Embeds protein sequences using the t5 model.

    Args:
        protein_sequence: list of protein sequences in single letter notation.

    Returns:
        embeddings: torch.Tensor of embeddings.
    """
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    logger.info("Loading t5 model.")
    tokenizer = T5Tokenizer.from_pretrained(
        "Rostlab/prot_t5_xl_half_uniref50-enc", do_lower_case=False
    )
    model = T5EncoderModel.from_pretrained("Rostlab/prot_t5_xl_half_uniref50-enc").to(
        device
    )

    # replace non-standard amino acids with X
    processed_sequences = [
        " ".join(list(re.sub(r"[UZOB]", "X", sequence)))
        for sequence in protein_sequences
    ]

    logger.info("Tokenizing protein sequences.")
    ids = tokenizer(processed_sequences, add_special_tokens=True, padding="longest")

    input_ids = torch.tensor(ids["input_ids"]).to(device)
    attention_mask = torch.tensor(ids["attention_mask"]).to(device)

    with torch.no_grad():
        embedding_repr = model(input_ids=input_ids, attention_mask=attention_mask)

    # embeddings shape [batch_size, len_longest_sequence + 1, hidden_size]
    # NOTE: sequences are still padded, can use attention_mask to remove padding
    # NOTE: EOS token is included in the embeddings (also needs to be removed)
    return embedding_repr.last_hidden_state


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

    with open(args.input_sequences, "r") as f:
        protein_sequences = f.readlines()

    embeddings = embed_protein_t5_xl(protein_sequences)
    save_tensor(embeddings=embeddings, output_dir=args.output_dir, type=args.type)
