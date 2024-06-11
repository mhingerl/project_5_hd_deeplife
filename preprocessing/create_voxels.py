import gzip
import shutil
import os
import pickle
from moleculekit.molecule import Molecule
from moleculekit.tools.voxeldescriptors import getVoxelDescriptors
from moleculekit.tools.atomtyper import prepareProteinForAtomtyping
import numpy as np
from tqdm import tqdm
from typing import Dict
from Bio.PDB import PDBParser, PDBIO, Select
from Bio.PDB.Polypeptide import is_aa


class ProteinSelect(Select):
    """Selects only residues belonging to a protein when parsing PDB files."""

    def accept_residue(self, residue):
        return is_aa(residue)


def load_and_save_protein(input_pdb, output_pdb):
    """Parses a PDB file and saves only the atoms belonging to amino acids."""
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("protein", input_pdb)
    io = PDBIO()
    io.set_structure(structure)
    io.save(output_pdb, ProteinSelect())


def unzip_file(file_path: str, output_path: str) -> None:
    """Unizps a file and saves the result to a given output path."""
    with gzip.open(file_path, "rb") as f_in:
        with open(output_path, "wb") as f_out:
            shutil.copyfileobj(f_in, f_out)


def voxelize_preprocessed_pdb(pdb_path: str) -> Dict[str, np.ndarray]:
    """Loads a PDB file and generates voxels with 8 feature channels."""
    mol = Molecule(pdb_path)
    mol = prepareProteinForAtomtyping(mol, verbose=False)
    features, centers, N = getVoxelDescriptors(mol)
    return {"features": features, "centers": centers, "N": N}


def voxelize_single_pdb(input_file, output_folder, error_log_file):
    if input_file.endswith(".pdb.gz") and not input_file.startswith("._"):
        pdb_path = os.path.join(input_folder, input_file)
        output_path = os.path.join(
            output_folder, f"{os.path.splitext(input_file)[0][:-4]}.pkl"
        )
        error_msg = ""

        try:
            pdb_out_path = pdb_path[:-3]
            unzip_file(file_path=pdb_path, output_path=pdb_out_path)
            load_and_save_protein(pdb_out_path, pdb_out_path)
            data = voxelize_preprocessed_pdb(pdb_out_path)

            with open(output_path, "wb") as f:
                pickle.dump(data, f)

        except Exception as e:
            error_msg = f"Error processing {input_file}: {str(e)}\n"
            with open(error_log_file, "a") as error_file:
                error_file.write(error_msg)


def process_pdb_files(input_folder, output_folder, error_log_file):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for i, filename in tqdm(enumerate(os.listdir(input_folder))):
        if filename.endswith(".pdb.gz") and not filename.startswith("._"):
            pdb_path = os.path.join(input_folder, filename)
            output_path = os.path.join(
                output_folder, f"{os.path.splitext(filename)[0][:-4]}.pkl"
            )
            error_msg = ""

            try:
                pdb_out_path = pdb_path[:-3]
                unzip_file(file_path=pdb_path, output_path=pdb_out_path)
                load_and_save_protein(pdb_out_path, pdb_out_path)
                data = voxelize_preprocessed_pdb(pdb_out_path)

                with open(output_path, "wb") as f:
                    pickle.dump(data, f)

            except Exception as e:
                error_msg = f"Error processing {filename}: {str(e)}\n"
                with open(error_log_file, "a") as error_file:
                    error_file.write(error_msg)


if __name__ == "__main__":
    pdb_dir = "/workspace/shared_data/pdb_files"
    for dirk in os.listdir(pdb_dir):
        input_folder = f"/workspace/shared_data/pdb_files/{dirk}"
        output_folder = f"/workspace/shared_data/pdb_files/{dirk}/out"
        error_log_file = os.path.join(output_folder, "error_log.txt")
        process_pdb_files(input_folder, output_folder, error_log_file)
