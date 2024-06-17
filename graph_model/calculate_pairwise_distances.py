from pathlib import Path
import random

from Bio import PDB, pairwise2
import numpy as np
from itertools import combinations
from torch_geometric.data import Data as GraphData
import torch


# Function to calculate distance between two points in 3D space
def calc_distance(coord1, coord2):
    return np.sqrt(np.sum((coord1 - coord2) ** 2))


# Function to extract CA atom coordinates
def get_ca_coords(pdb_file, chain_id):
    parser = PDB.PDBParser(QUIET=True)
    structure = parser.get_structure("protein", pdb_file)
    ca_coords = []
    for model in structure:
        for chain in model:
            if chain.id == chain_id:
                for residue in chain:
                    if residue.id[0] == " ":
                        atom_coords = []
                        for atom in residue:
                            atom_coords.append(atom.get_coord())
                        atom_coords = np.array(atom_coords).mean(axis=0)
                        ca_coords.append(atom_coords)
    return np.array(ca_coords)


# Function to calculate pairwise distances
def calculate_pairwise_distances(ca_coords):
    num_atoms = len(ca_coords)
    distance_matrix = np.zeros((num_atoms, num_atoms))
    for i, j in combinations(range(num_atoms), 2):
        distance = calc_distance(ca_coords[i], ca_coords[j])
        distance_matrix[i, j] = distance
        distance_matrix[j, i] = distance  # distance is symmetric
    return distance_matrix


def sample_pairs(distance_matrix, num_samples):
    # Inverse the distances and flatten the upper triangle of the matrix
    upper_tri_indices = np.triu_indices_from(distance_matrix, k=1)
    distances = distance_matrix[upper_tri_indices]
    inverse_distances = 1 / distances
    probabilities = inverse_distances / np.sum(inverse_distances)

    # Create pairs
    pairs = list(zip(upper_tri_indices[0], upper_tri_indices[1]))

    # Sample pairs
    sampled_pairs = random.choices(pairs, weights=probabilities, k=num_samples)

    return sampled_pairs


# Main function
def main(pdb_file, chain_id, num_samples):
    ca_coords = get_ca_coords(pdb_file, chain_id=chain_id)
    distance_matrix = calculate_pairwise_distances(ca_coords)
    sampled_pairs = sample_pairs(distance_matrix, num_samples)
    return sampled_pairs, ca_coords.shape[0]


import hashlib
import json
from loguru import logger
from tqdm import tqdm

import numpy as np
from Bio import pairwise2


def get_pdb_info_search_idx(extracted_info) -> dict:
    search_index = dict()
    for i, entry in enumerate(extracted_info):
        search_index[f"{entry['pdb_id']}_{entry['chain']}"] = i
    return search_index


def get_pocket_residues(entry, chain, mode="holo") -> list:
    pocket_ids = [resid.split("_") for resid in entry[f"{mode}_pocket_selection"]]
    pocket_ids = [resid[1] for resid in pocket_ids if resid[0] == chain]
    return pocket_ids


def process_sequence_graph_dataset(split_annotations, extracted_info, embeddings_path, fasta_path, pdb_path, out_dir, mode="holo"):
    Xs = {}
    search_idx = get_pdb_info_search_idx(extracted_info)
    fake_data_point_counter = 0

    for apo_pdb_id, holo_entries in tqdm(split_annotations.items()):
        for holo_entry in holo_entries:
            holo_pdb_id = holo_entry["holo_pdb_id"]
            holo_chain = holo_entry["holo_chain"]
            apo_chain = holo_entry["apo_chain"]
            uniprot_id = holo_entry["uniprot_id"]

            pocket_chain = holo_chain if mode == "holo" else apo_chain
            pocket_ids = get_pocket_residues(holo_entry, pocket_chain, mode=mode)

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

            # NOTE: one more edge case, lol
            if fasta_sequence == "":
                logger.warning(f"Fasta {fasta_file} is empty.")
                continue

            # load embedding
            embedding_file = embeddings_path / f"{seq_id_string}.npy"
            embedding = np.load(embedding_file)

            # load extracted pdb seqs
            try:
                pdb_id = holo_pdb_id if mode == "holo" else apo_pdb_id
                chain = holo_chain if mode == "holo" else apo_chain
                entry_id = search_idx[f"{pdb_id}_{chain}"]
            except KeyError:
                continue
                # logger.error(f"pdb file {holo_pdb_id} does not exist :(")

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
                pdb_file_path = pdb_path / f"{pdb_id}.pdb"
                ca_coords = get_ca_coords(pdb_file_path, chain_id=chain)
                try:
                    ca_coords = ca_coords[masked_overlap]
                except IndexError:
                    logger.error(f"Failed to extract CA coords for {pdb_id} {chain}")
                    continue
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

            distance_matrix = calculate_pairwise_distances(ca_coords)
            sampled_pairs = sample_pairs(distance_matrix, 100)
            sequence_edges = torch.tensor(
                [[i, i+1] for i in range(len(filtered_embedding)-1)]
            )
            full_edges = torch.cat([sequence_edges, torch.tensor(sampled_pairs)], dim=0)
            graph_data = generate_graph_dataset(
                node_features=torch.tensor(filtered_embedding, dtype=torch.float),
                edge_index=full_edges,
                node_labels=torch.tensor(labels, dtype=torch.long)
            )

            # store as json
            out_dir.mkdir(parents=True, exist_ok=True)
            graph_dict = graph_data.to_dict()
            graph_filename = out_dir / f"{holo_pdb_id}_{chain}_{checksum}.gz"
            save_compressed_tensor_dict(graph_dict, graph_filename)

            assert filtered_embedding.shape[0] == labels.shape[0] == ca_coords.shape[0]

            Xs[checksum] = graph_data

    # logger.info(f"Fake data points: {fake_data_point_counter}")
    return Xs


def generate_graph_dataset(node_features, edge_index, node_labels, undirected=True):
    edge_index = edge_index.t().contiguous()

    if undirected:
        edge_index = torch.cat([edge_index, edge_index[[1, 0]]], dim=-1)

    data = GraphData(x=node_features, edge_index=edge_index, y=node_labels)
    return data

import os
import pickle
import gzip
def save_compressed_tensor_dict(tensor_dict, filename):
    with open('temp_tensor_dict.pkl', 'wb') as f:
        pickle.dump(tensor_dict, f, protocol=pickle.HIGHEST_PROTOCOL)

    with open('temp_tensor_dict.pkl', 'rb') as f_in:
        with gzip.open(filename, 'wb') as f_out:
            f_out.writelines(f_in)
    
    os.remove('temp_tensor_dict.pkl')


def load_compressed_tensor_dict(filename):
    with gzip.open(filename, 'rb') as f_in:
        with open('temp_tensor_dict_out.pkl', 'wb') as f_out:
            f_out.writelines(f_in)
    
    with open('temp_tensor_dict_out.pkl', 'rb') as f:
        tensor_dict = pickle.load(f)
    
    os.remove('temp_tensor_dict_out.pkl')
    
    return tensor_dict


if __name__ == "__main__":
    modes = ["apo", "holo"]
    files = ["train/train-fold-0.json", "train/train-fold-1.json", "train/train-fold-2.json", "train/train-fold-3.json", "train/train-fold-4.json", "test/test.json"]
    from pathlib import Path
    from itertools import product

    for file, mode in product(files, modes):
        with open(Path("/workspace/CryptoBench/single_chain") / file) as f:
            split_annotations = json.load(f)

        with open(f"/workspace/pdb_bullshit/extracted_sequences/{mode}_sc_seqs.json") as f:
            extracted_info = json.load(f)

        Xs_train = process_sequence_graph_dataset(
            split_annotations=split_annotations,
            extracted_info=extracted_info,
            embeddings_path=Path("/workspace/uniprot_embeddings/ankh/sc"),
            fasta_path=Path("/workspace/fastas/sc"),
            pdb_path=Path(f"/workspace/pdb_bullshit/pdb_files/{mode}_sc"),
            out_dir=Path(f"/workspace/graphs/{mode}/sc/{Path(file).stem}"),
            mode=mode
        )
        logger.info(f"Train set: {len(Xs_train)}")




