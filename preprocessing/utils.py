from Bio.PDB import parse_pdb_header
import gzip
import shutil


def unzip_file(file_path: str, output_path: str) -> None:
    """Unizps a file and saves the result to a given output path."""
    with gzip.open(file_path, "rb") as f_in:
        with open(output_path, "wb") as f_out:
            shutil.copyfileobj(f_in, f_out)


def extract_chains(pdb_file: str) -> list[str]:
    """
    Extracts chains from pdb file.

    Args:
        pdb_file: str of pdb file.

    Returns:
        chains: list of chain ids.
    """
    header = parse_pdb_header(pdb_file)
    # this hurts...
    compounds = header["compound"]

    chains = []
    for _, entry in compounds.items():
        chains.extend(entry["chain"].upper().strip().split(", "))
    return chains
