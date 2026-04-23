import os
import os.path as osp
import ssl
from tqdm import tqdm
import urllib
from typing import Optional


def _download_from_mirrors(
    *,
    repo: str,
    filename: str,
    download_path: str,
) -> bool:
    urls = [
        f"https://huggingface.co/{repo}/resolve/main/{filename}?download=true",
        f"https://hf-mirror.com/{repo}/resolve/main/{filename}?download=true",
    ]
    for url in urls:
        try:
            download_url(url=url, folder=download_path, filename=filename)
            print(f"Downloaded successfully from {url}")
            return True
        except Exception as e:  # noqa: BLE001
            print(f"Failed to download from {url}: {e}")
    return False


def _download_direct_url(
    *,
    url: str,
    filename: str,
    download_path: str,
) -> bool:
    try:
        download_url(url=url, folder=download_path, filename=filename)
        print(f"Downloaded successfully from {url}")
        return True
    except Exception as e:  # noqa: BLE001
        print(f"Failed to download from {url}: {e}")
        return False


def download_url(
    url: str,
    folder: str,
    filename: Optional[str] = None,
):
    r"""Downloads the content of an URL to a specific folder.

    Args:
        url (str): The URL.
        folder (str): The folder.
        log (bool, optional): If :obj:`False`, will not print anything to the
            console. (default: :obj:`True`)
        filename (str, optional): The filename of the downloaded file. If set
            to :obj:`None`, will correspond to the filename given by the URL.
            (default: :obj:`None`)
    Returns:
        path (str): Path of the contents downloaded.
    """
    if filename is None:
        filename = url.rpartition("/")[2]
        filename = filename if filename[0] == "?" else filename.split("?")[0]

    path = osp.join(folder, filename)
    os.makedirs(folder, exist_ok=True)
    # context = ssl._create_unverified_context()
    context = ssl.create_default_context()
    # safe check:
    assert url[:4].lower() == "http", "Only HTTP or HTTPS is supported."
    data = urllib.request.urlopen(url, context=context)

    with open(path, "wb") as f, tqdm(
        desc=f"Downloading {url}",
        total=int(data.info().get("Content-Length", -1)),
        unit="B",
        unit_scale=True,
        unit_divisor=1024,
    ) as bar:
        while True:
            chunk = data.read(10 * 1024 * 1024)
            if not chunk:
                break
            size = f.write(chunk)
            bar.update(size)
    return path


def download_google_url(
    id: str,
    folder: str,
    filename: str,
):
    r"""Downloads the content of a Google Drive ID to a specific folder.

    Args:

        id (str): Google Drive ID.
        folder (str): The folder.
        filename (str): The filename of the downloaded file.

    """
    url = f"https://drive.usercontent.google.com/download?id={id}&confirm=t"
    return download_url(url, folder, filename)


def download_model_from_huggingface(
    repo: str,
    model_name: str,
    download_path: str,
) -> bool:
    """Download a model checkpoint from primary/fallback mirrors."""
    os.makedirs(download_path, exist_ok=True)
    model_ok = _download_from_mirrors(
        repo=repo,
        filename=model_name,
        download_path=download_path,
    )
    if not model_ok:
        print("Download failed from both URLs.")
        return False

    companion_files: list[str] = []
    if "tabpfn-v2.6-" in model_name:
        companion_files.append("pre_generated_column_embeddings_v2_6.pt")

    for companion in companion_files:
        companion_path = osp.join(download_path, companion)
        if osp.exists(companion_path):
            continue
        companion_ok = _download_from_mirrors(
            repo=repo,
            filename=companion,
            download_path=download_path,
        )
        if not companion_ok:
            print(
                f"Optional companion file was not downloaded: {companion}. "
                "The checkpoint download itself succeeded."
            )

    column_embedding_filename = "pre_generated_column_embeddings_v2_6.pt"
    column_embedding_path = osp.join(download_path, column_embedding_filename)
    if not osp.exists(column_embedding_path):
        column_embedding_ok = _download_direct_url(
            url=(
                "https://raw.githubusercontent.com/PriorLabs/TabPFN/main/"
                "src/tabpfn/architectures/shared/tabpfn_col_embedding.pt"
            ),
            filename=column_embedding_filename,
            download_path=download_path,
        )
        if not column_embedding_ok:
            print(
                "Optional companion file was not downloaded: "
                "pre_generated_column_embeddings_v2_6.pt. "
                "The checkpoint download itself succeeded."
            )

    return True
