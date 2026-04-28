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


def download_url(
    url: str,
    folder: str,
    filename: Optional[str] = None,
):
    r"""Download the content of a URL to a specific folder.

    Args:
        url (str): The URL to download from.
        folder (str): The destination folder.
        filename (str, optional): The filename of the downloaded file. If set
            to :obj:`None`, the filename is inferred from the URL.
            (default: :obj:`None`)

    Returns:
        str: The local path to the downloaded file.
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
    r"""Download the content of a Google Drive file to a specific folder.

    Args:
        id (str): The Google Drive file ID.
        folder (str): The destination folder.
        filename (str): The filename of the downloaded file.

    Returns:
        str: The local path to the downloaded file.
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

    return True
