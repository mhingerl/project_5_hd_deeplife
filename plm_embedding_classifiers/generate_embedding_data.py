from itertools import product
from pathlib import Path

from linear_model import save_sequence_dataset


if __name__ == "__main__":
    modes = ["apo", "holo"]
    models = ["ankh", "t5"]
    files = [
        "train/train-fold-0.json",
        "train/train-fold-1.json",
        "train/train-fold-2.json",
        "train/train-fold-3.json",
        "train/train-fold-4.json",
        "test/test.json",
    ]

    for mode, model, file in product(modes, models, files):

        save_sequence_dataset(
            dataset_json=file,
            extracted_sequences_json=f"{mode}_sc_seqs.json",
            model=model,
            mode=mode,
            save_path=Path(f"/workspace/sequence_datasets/{model}/{mode}"),
            file_prefix=Path(file).stem,
        )
