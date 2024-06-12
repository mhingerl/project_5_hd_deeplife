"""
Utility file to embed protein data using ankh.

Run as script by providing a file with protein sequences separated by \\n.

Example:
    python ankh.py --input_sequences data.json --output_dir embeddings --type numpy
"""

import argparse
import gc
from pathlib import Path
from tqdm import tqdm
import re

from loguru import logger
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from transformers import T5EncoderModel, T5Tokenizer


from utils import subset_embeddings_with_attention_mask, ProteinDataset


def toknize_ankh(tokenizer, sequences):
    """Tokenizes protein sequences using ankh tokenizer."""
    protein_sequences = [list(seq) for seq in sequences]
    outputs = tokenizer.batch_encode_plus(
        protein_sequences,
        add_special_tokens=True,
        padding=True,
        is_split_into_words=True,
        return_tensors="pt",
    )
    return outputs["input_ids"], outputs["attention_mask"]


def tokenize_t5(tokenizer, sequences):
    """Tokenizes protein sequences using t5 tokenizer."""
    processed_sequences = [
        " ".join(list(re.sub(r"[UZOB]", "X", sequence))) for sequence in sequences
    ]
    outputs = tokenizer(
        processed_sequences,
        add_special_tokens=True,
        padding="longest",
        return_tensors="pt",
    )
    return outputs["input_ids"], outputs["attention_mask"]


def load_model(model_name):
    """Loads the model and tokenizer from HuggingFace."""
    if model_name == "ankh":
        hf_model = "ElnaggarLab/ankh-large"
        tokenizer = AutoTokenizer.from_pretrained(hf_model)
    elif model_name == "t5":
        hf_model = "Rostlab/prot_t5_xl_half_uniref50-enc"
        tokenizer = T5Tokenizer.from_pretrained(hf_model, do_lower_case=False)
    else:
        raise ValueError(f"Model {model_name} not supported. Use either ankh or t5.")
    logger.info(f"Loading {model_name} model.")
    model = T5EncoderModel.from_pretrained(hf_model)
    return tokenizer, model


def embed_protein(model, tokenizer, data_loader: dict[str, str], model_name: str):
    """
    Embeds protein sequences using ankh model.

    Args:
        protein_sequence: list of protein sequences in single letter notation.

    Returns:
        embeddings: torch.Tensor of embeddings.
    """
    for i, (filenames, sequences) in enumerate(data_loader):
        if model_name == "ankh":
            input_ids, attention_mask = toknize_ankh(tokenizer, sequences)
        elif model_name == "t5":
            input_ids, attention_mask = tokenize_t5(tokenizer, sequences)
        else:
            raise ValueError(
                f"Model {model_name} not supported. Use either ankh or t5."
            )

        logger.info(f"Embedding protein batch {i+1}/{len(data_loader)}.")
        with torch.no_grad():
            embeddings = model(
                input_ids=input_ids.to(device),
                # attention mask [batch_size, max_seq_len + 1]
                attention_mask=attention_mask.to(device),
            )

        # embeddings [batch_size, len_longest_sequence + 1, hidden_size]
        embeddings = embeddings.last_hidden_state
        final_embeddings = subset_embeddings_with_attention_mask(
            embeddings, attention_mask
        )

        del input_ids, attention_mask, embeddings
        gc.collect()
        torch.cuda.empty_cache()

        yield zip(filenames, final_embeddings)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--input_sequences", type=str, help="Path to input file.")
    parser.add_argument("--output_dir", type=str, help="Path to output directory.")
    parser.add_argument("--model", type=str, help="Model to use (ankh or t5).")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size.")

    args = parser.parse_args()

    logger.info(f"Reading protein sequences from {args.input_sequences}.")
    df = pd.read_csv(args.input_sequences)

    sequences = dict()
    for protein_id, protein in df.iterrows():
        protein.dropna(inplace=True)
        protein = protein.to_dict()
        for chain, sequence in protein.items():
            sequences[f"{protein_id}_{chain}"] = sequence

    protein_loader = DataLoader(
        ProteinDataset(sequences), batch_size=args.batch_size, shuffle=False
    )

    tokenizer, model = load_model(args.model)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    embed_batches = embed_protein(model, tokenizer, protein_loader, args.model)

    out_path = Path(args.output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    for batch in embed_batches:
        for filename, emb in batch:
            np.save(out_path / f"{filename}.npy", emb)
