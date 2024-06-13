"""
Extracts the sequence from a (zipped) pdb file.
"""

import argparse
import json
from pathlib import Path

from Bio.PDB import PDBParser
from Bio.SeqUtils import seq1
from loguru import logger

from utils import unzip_file


def extract_protein_sequence(pdb_file):
    parser = PDBParser()
    structure = parser.get_structure("PDB_structure", pdb_file)

    protein_sequences = []

    pdb_id = pdb_file.split("/")[-1][:-4]

    for model in structure:
        for chain in model:
            chain_id = chain.id
            sequence = []
            residue_ids = []

            for residue in chain:
                if (
                    residue.id[0] == " "
                ):  # Ensures only standard residues (ignores HETATM)
                    res_id = residue.id[1]
                    res_name = seq1(residue.resname)
                    sequence.append(res_name)
                    residue_ids.append(res_id)

            protein_sequences.append(
                {
                    "pdb_id": pdb_id,
                    "chain": chain_id,
                    "sequence": "".join(sequence),
                    "residue_ids": residue_ids,
                }
            )

    return protein_sequences


def extract_sequence(pdb_path: str) -> dict[str, str]:
    """Loads a PDB file and generates voxels with 8 feature channels."""
    pdb_out_path = pdb_path[:-3]
    unzip_file(file_path=pdb_path, output_path=pdb_out_path)
    chain_sequences = extract_protein_sequence(pdb_out_path)
    return chain_sequences


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extracts the sequence from a (zipped) pdb file."
    )
    parser.add_argument("--pdb_path", type=str, help="Path to the pdb file.")
    parser.add_argument(
        "--output_path",
        type=str,
        help="Path to the output folder where the sequence will be saved.",
    )
    args = parser.parse_args()

    seqs = []
    for pdb_path in Path(args.pdb_path).rglob("*.pdb.gz"):
        # # TODO: rm only for debugging
        if not pdb_path.stem.startswith("1"):
            continue
        if pdb_path.stem.startswith("._"):
            continue
        logger.info(f"Extracting sequence from {pdb_path}")
        # try:
        chain_sequences = extract_sequence(str(pdb_path))
        seqs.extend(chain_sequences)
        # except Exception as e:
        # logger.error(f"Failed to extract sequence from {pdb_path}: {e}")

    with open(Path(args.output_path), "w") as f:
        json.dump(seqs, f)
