import gzip
import shutil


def unzip_file(file_path: str, output_path: str) -> None:
    """Unizps a file and saves the result to a given output path."""
    with gzip.open(file_path, "rb") as f_in:
        with open(output_path, "wb") as f_out:
            shutil.copyfileobj(f_in, f_out)
