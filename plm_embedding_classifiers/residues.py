import hashlib
from tqdm import tqdm
from loguru import logger

import numpy as np
import json

from Bio import pairwise2


def get_pdb_info_search_idx(extracted_info) -> dict:
    search_index = dict()
    for i, entry in enumerate(extracted_info):
        search_index[f"{entry['pdb_id']}_{entry['chain']}"] = i
    return search_index


def get_pocket_residues(entry, chain) -> list:
    pocket_ids = [resid.split("_") for resid in entry["holo_pocket_selection"]]
    pocket_ids = [resid[1] for resid in pocket_ids if resid[0] == chain]
    return pocket_ids


def process_sequence_dataset(split_annotations, extracted_info, embeddings_path, fasta_path):
    Xs = {}
    Ys = {}
    search_idx = get_pdb_info_search_idx(extracted_info)
    fake_data_point_counter = 0

    for apo_pdb_id, holo_entries in tqdm(split_annotations.items()):
        for holo_entry in holo_entries:
            holo_pdb_id = holo_entry["holo_pdb_id"]
            holo_chain = holo_entry["holo_chain"]
            apo_chain = holo_entry["apo_chain"]
            uniprot_id = holo_entry["uniprot_id"]
            pocket_ids = get_pocket_residues(holo_entry, holo_chain)

            identifier = f"{apo_pdb_id}_{uniprot_id}_{"".join(pocket_ids)}"
            result = hashlib.md5(identifier.encode())
            checksum = result.hexdigest()

            # TODO: load fastas correctly and reembed
            # NOTE: this marks an edge case to not do this right now! :)
            if len(apo_chain) > 1:
                continue
            # NOTE: this is also an edge case, who the fuck makes such stupid file conventions???!!!
            if len(holo_chain) > 1:
                continue

            fake_data_point_counter += 1

            if Xs.get(checksum, None) is not None:
                continue

            seq_id_string = f"{apo_pdb_id}_{apo_chain}_{uniprot_id}"
            # load fasta sequence
            fasta_file = fasta_path / f"{seq_id_string}.fasta"
            with open(fasta_file) as f:
                content = f.readlines()
                fasta_sequence = "".join([line.strip() for line in content[1:]])

            # load embedding
            embedding_file = embeddings_path / f"{seq_id_string}.npy"
            embedding = np.load(embedding_file)

            # load extracted pdb seqs
            try:
                entry_id = search_idx[f"{holo_pdb_id}_{holo_chain}"]
            except KeyError:
                logger.error(f"pdb file {holo_pdb_id} does not exist :(")

            pdb_info = extracted_info[entry_id]

            # match seq with fasta
            alignments = pairwise2.align.globalms(
                fasta_sequence, pdb_info["sequence"], 1, 0, -0.5, -0.1
            )
            aligned_fasta = alignments[0].seqA
            aligned_pdb_sequence = alignments[0].seqB

            # NOTE: edgecase pdb_seq is longer than fasta sequence
            overlap = np.where(np.array(list(aligned_fasta)) == "-", False, True)
            if len(overlap) > 0:
                aligned_pdb_sequence = "".join(np.array(list(aligned_pdb_sequence))[overlap])
                overlap_mask = np.array(
                    [False if aa == "-" else True for aa in alignments[0].seqB]
                )
                masked_overlap = overlap[overlap_mask]
                residue_ids = np.array(pdb_info["residue_ids"])[masked_overlap]
            else:
                # create labels using entry residue ids and extracted pdb seq ids
                residue_ids = pdb_info["residue_ids"]

            # filter embeddings
            embeding_mask = np.array(
                [False if aa == "-" else True for aa in aligned_pdb_sequence]
            )
            filtered_embedding = embedding[embeding_mask]

            labels = np.array(
                [1 if resid in pocket_ids else 0 for resid in residue_ids]
            )

            assert filtered_embedding.shape[0] == labels.shape[0]

            Xs[checksum] = filtered_embedding
            Ys[checksum] = labels

    logger.info(f"Fake data points: {fake_data_point_counter}")
    return Xs, Ys


if __name__ == "__main__":
    from pathlib import Path
    with open("/workspace/CryptoBench/single_chain/train/train-fold-0.json") as f:
        split_annotations = json.load(f)

    with open("/workspace/pdb_bullshit/extracted_sequences/holo_sc_seqs.json") as f:
        extracted_info = json.load(f)

    Xs_train, Ys_train = process_sequence_dataset(
        split_annotations=split_annotations,
        extracted_info=extracted_info,
        embeddings_path=Path("/workspace/uniprot_embeddings/ankh/sc"),
        fasta_path=Path("/workspace/fastas/sc"),
    )
    logger.info(f"Train set: {len(Xs_train)}")

