import numpy as np
import json
import os


def process_sequence_dataset(annotation_path, embeddings_path):
    with open(f"{annotation_path}") as f:
        proteins = json.load(f)

        Xs = {}
        Ys = {}

        for protein, lists in proteins.items():
            names2 = []
            for list in lists:
                names = [f"{protein}_{x}" for x in list["apo_pocket_selection"]]
                names2 = names2.extend(names)
                
            apo_pocket = proteins[protein]
            ligands = proteins[protein]
            chains = []
            for ligand in ligands:

                chains = chains.extend()







                for chain in chains:
                    protein_chain = f"{protein}_{chain}"
                    if protein_chain not in Xs:
                            filename = protein_chain + ".npy"
                            embedding = np.load(f"{embeddings_path}/{filename}")
                            Xs[protein_chain] = embedding

                    if protein_chain not in Ys:
                        Ys[protein_chain] = np.zeros(embedding.shape[0])

                    for ligand in proteins[protein]["apo_pocket_selection"]:
                        for residue in [residue for residue in ligand["apo_pocket_selection"]]:
                            if residue[0] == protein[-1]:
                                Ys[protein][int(residue[2:]) - 1] = 1



    filenames = os.listdir(embeddings_path)
    proteins = [filename[:-4] for filename in filenames]

        return Xs, Ys


if __name__ == "__main__":

    Xs_train, Ys_train = process_sequence_dataset(
        "./CryptoBench/single_chain/whole_dataset.json",
        "./pdb_bullshit/ankh_embeddings/",
    )
