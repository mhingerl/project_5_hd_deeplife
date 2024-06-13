"""
Utility file to embed protein data using ankh.

Run as script by providing a file with protein sequences separated by \\n.

Example:
    python ankh.py --input_sequences data.json --output_dir embeddings --type numpy
"""

import argparse
import gc
import json
from pathlib import Path
from tqdm import tqdm
import re
from collections import defaultdict

from loguru import logger
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from transformers import T5EncoderModel, T5Tokenizer


from utils import subset_embeddings_with_attention_mask, ProteinDataset


def tokenize_ankh(tokenizer, sequences):
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


def chunk_sequence(sequence, chunk_size, stride):
    """
    Splits a sequence into overlapping chunks.
    
    Args:
        sequence (str): The protein sequence.
        chunk_size (int): The size of each chunk.
        stride (int): The number of residues to move between chunks.
        
    Returns:
        list: A list of sequence chunks.
    """
    chunk_info = []
    for i in range(0, len(sequence), stride):
        start_idx = i
        end_idx = i + chunk_size
        if end_idx > len(sequence):
            end_idx = len(sequence)
        chunk = sequence[start_idx : end_idx]
        chunk_info.append(
            (start_idx, end_idx, chunk)
        )
    return chunk_info


def embed_protein(model, tokenizer, data_loader: dict[str, str], model_name: str):
    """
    Embeds protein sequences using ankh model.

    Args:
        protein_sequence: list of protein sequences in single letter notation.

    Returns:
        embeddings: torch.Tensor of embeddings.
    """
    for i, data in enumerate(data_loader):
        chunked_data = []
        raw_chunks = []
        data = zip(*data)
        for filename, sequence in data:
            chunk_info = chunk_sequence(sequence, 512, 256)
            for start_idx, end_idx, chunk in chunk_info:
                chunked_data.append((filename, start_idx, end_idx, len(sequence)))
                raw_chunks.append(chunk)

        if model_name == "ankh":
            input_ids, attention_mask = tokenize_ankh(tokenizer, raw_chunks)
        elif model_name == "t5":
            input_ids, attention_mask = tokenize_t5(tokenizer, raw_chunks)
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

        averaged_embeddings = defaultdict(list)
        for i, (filename, start_idx, end_idx, seq_len) in enumerate(chunked_data):
            # [seq_len, hidden_size]
            chunk_embed = np.full((seq_len, final_embeddings[i].shape[1]), np.nan)
            chunk_embed[start_idx:end_idx] = final_embeddings[i]
            averaged_embeddings[filename].append(chunk_embed)

        final_embeddings = []
        filenames = []
        for filename, chunks in averaged_embeddings.items():
            emb = np.stack(chunks, axis=0)
            emb = np.nanmean(emb, axis=0)
            final_embeddings.append(emb)
            filenames.append(filename)
            
        input_ids.detach().cpu()
        attention_mask.detach().cpu()
        embeddings.detach().cpu()
        del input_ids, attention_mask, embeddings
        gc.collect()
        torch.cuda.empty_cache()

        yield zip(filenames, final_embeddings)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--input_sequences", type=str, default="pdb_bullshit/extracted_sequences/apo_sc_seqs.json", help="Path to input file.")
    parser.add_argument("--output_dir", type=str, default="pdb_bullshit/ankh_test_embeddings", help="Path to output directory.")
    parser.add_argument("--model", type=str, default="ankh", help="Model to use (ankh or t5).")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size.")
    parser.add_argument("--mode", type=str, default="json", help="Mode to read input file (json or fasta).")

    args = parser.parse_args()

    logger.info(f"Reading protein sequences from {args.input_sequences}.")

    model_input = dict()
    if args.mode == "json":
        with open(args.input_sequences) as f:
            sequences = json.load(f)

        for protein_chain in sequences:
            model_input[f"{protein_chain["pdb_id"]}_{protein_chain["chain"]}"] = protein_chain["sequence"]
    elif args.mode == "fasta":
        for file in Path(args.input_sequences).rglob("*.fasta"):
            with open(file) as f:
                content = f.readlines()
                model_input[file.stem] = "".join([line.strip() for line in content[1:]])

    protein_loader = DataLoader(
        ProteinDataset(model_input), batch_size=args.batch_size, shuffle=False
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
