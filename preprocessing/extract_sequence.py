"""
Extracts the sequence from a (zipped) pdb file.
"""

import argparse
from pathlib import Path

from loguru import logger
from moleculekit.molecule import Molecule
import pandas as pd

from utils import unzip_file


def extract_sequence(pdb_path: str) -> dict[str, str]:
    """Loads a PDB file and generates voxels with 8 feature channels."""
    pdb_out_path = pdb_path[:-3]
    unzip_file(file_path=pdb_path, output_path=pdb_out_path)
    mol = Molecule(pdb_out_path)
    return mol.sequence()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extracts the sequence from a (zipped) pdb file."
    )
    parser.add_argument("pdb_path", type=str, help="Path to the pdb file.")
    parser.add_argument(
        "--output_path",
        type=str,
        help="Path to the output folder where the sequence will be saved.",
    )
    args = parser.parse_args()

    seqs = []
    for pdb_path in Path(args.pdb_path).rglob("*.pdb"):
        logger.info(f"Extracting sequence from {pdb_path}")
        sequence = extract_sequence(args.pdb_path)
        result = dict(pdb_id=pdb_path.stem)
        result.update(sequence)
        seqs.append(result)

    df = pd.DataFrame(seqs)
    df.to_csv(args.output_path, index=False)
