"""
Extracts the sequence from a (zipped) pdb file.
"""

import argparse
from pathlib import Path

from loguru import logger
from moleculekit.molecule import Molecule
import numpy as np
import pandas as pd

from utils import unzip_file, extract_chains


def extract_protein_sequence(pdb_file):
    parser = PDBParser()
    structure = parser.get_structure("PDB_structure", pdb_file)

    protein_sequences = []

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
    mol = Molecule(pdb_out_path, validateElements=False)
    seq = mol.sequence()
    chains = extract_chains(pdb_out_path)
    mol_chains = np.unique(mol.chain)
    chains = [chain for chain in chains if chain in mol_chains]
    mapped_sequence = {}
    for i, chain in enumerate(chains):
        mapped_sequence[chain] = seq[str(i)]
    return mapped_sequence

    return {
        "sequence": seq,
        "chain": chain,
        "residue_ids": residue_ids,
    }


if __name__ == "__main__":
    # parser = argparse.ArgumentParser(
    #     description="Extracts the sequence from a (zipped) pdb file."
    # )
    # parser.add_argument("--pdb_path", type=str, help="Path to the pdb file.")
    # parser.add_argument(
    #     "--output_path",
    #     type=str,
    #     help="Path to the output folder where the sequence will be saved.",
    # )
    # args = parser.parse_args()

    # seqs = []
    # for pdb_path in Path(args.pdb_path).rglob("*.pdb.gz"):
    #     # # TODO: rm only for debugging
    #     # if not pdb_path.stem.startswith("1"):
    #     #     continue
    #     if pdb_path.stem.startswith("._"):
    #         continue
    #     logger.info(f"Extracting sequence from {pdb_path}")
    #     try:
    #         sequence = extract_sequence(str(pdb_path))
    #         result = dict(pdb_id=pdb_path.stem[:-4])
    #         result.update(sequence)
    #         seqs.append(result)
    #     except Exception as e:
    #         logger.error(f"Failed to extract sequence from {pdb_path}: {e}")

    # df = pd.DataFrame(seqs)
    # df.to_csv(args.output_path, index=False)

    from Bio.PDB import PDBParser
    from Bio.SeqUtils import seq1

    def extract_protein_sequence(pdb_file):
        parser = PDBParser()
        structure = parser.get_structure("PDB_structure", pdb_file)

        protein_sequence = []

        for model in structure:
            for chain in model:
                chain_id = chain.id
                for residue in chain:
                    if (
                        residue.id[0] == " "
                    ):  # Ensures only standard residues (ignores HETATM)
                        res_id = residue.id[1]
                        res_name = seq1(residue.resname)
                        protein_sequence.append((chain_id, res_id, res_name))

        return protein_sequence

    seq = extract_protein_sequence("/workspace/pdb_bullshit/apo_sc/1wyc.pdb")
    print(seq)
