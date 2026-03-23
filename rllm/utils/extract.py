import zipfile


def extract_zip(path: str, output_folder: str):
    r"""Extract a zip archive to a specific folder.

    Args:
        path (str): Path to the zip file.
        output_folder (str): Destination folder for the extracted contents.
    """
    with zipfile.ZipFile(path, "r") as f:
        f.extractall(output_folder)
