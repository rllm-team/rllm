import zipfile


def extract_zip(path: str, output_folder: str):
    r"""Extract a zip archive to a specific folder."""
    with zipfile.ZipFile(path, "r") as f:
        f.extractall(output_folder)
