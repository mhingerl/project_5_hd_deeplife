"""
This script downloads the FASTA sequences for the proteins in the CryptoBench whole dataset file.

Example usage:
    python download_uniprot_fastas.py --annotation_path ./CryptoBench/single_chain/whole_dataset.json --fasta_output_path ./CryptoBench/single_chain/fasta_files
"""

import argparse
import json
import numpy as np
import requests
from tqdm import tqdm


def get_uniprot_sequence(uniprot_id):
    """Queries Uniprot to retrieve the sequence for a given UniProt ID."""
    url = f"https://www.uniprot.org/uniprot/{uniprot_id}.fasta"
    response = requests.get(url)
    if response.status_code == 200:
        fasta_data = response.text
        return fasta_data
    else:
        print(f"Error fetching data for UniProt ID {uniprot_id}")
        return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotation_path", type=str, help="Path to annotation file.")
    parser.add_argument(
        "--fasta_output_path", type=str, help="Path to output directory."
    )

    args = parser.parse_args()

    uniprotID_list = []
    with open(f"{args.annotation_path}") as f:

        uniprotIDs = ""
        proteins = json.load(f)

        for protein in tqdm(proteins.keys()):
            sections = proteins[protein]
            for section in sections:
                residues = section["apo_pocket_selection"]
                chains = [residue.split("_")[0] for residue in residues]
                unique_chains = np.unique(chains)

                if uniprotIDs != section["uniprot_id"]:
                    uniprotIDs = section["uniprot_id"]
                    uniprotID_list = uniprotIDs.split("-")
                    for uniprotID in uniprotID_list:
                        fasta = get_uniprot_sequence(uniprotID)

                        for unique_chain in unique_chains:
                            if len(uniprotID_list) > 1:
                                filename = f"{args.fasta_output_path}/{protein}_{unique_chain}_{uniprotID}_mc.fasta"
                            else:
                                filename = f"{args.fasta_output_path}/{protein}_{unique_chain}_{uniprotID}.fasta"
                            with open(filename, "w") as file:
                                file.write(fasta)
