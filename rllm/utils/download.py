import os
import os.path as osp
import ssl
from tqdm import tqdm
import urllib
from typing import Optional


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
        filename = url.rpartition('/')[2]
        filename = filename if filename[0] == '?' else filename.split('?')[0]

    path = osp.join(folder, filename)
    os.makedirs(folder, exist_ok=True)
    # context = ssl._create_unverified_context()
    context = ssl.create_default_context()
    # safe check:
    assert url[:4].lower() == "http", 'Only HTTP or HTTPS is supported.'
    data = urllib.request.urlopen(url, context=context)

    with open(path, 'wb') as f, tqdm(
        desc=f'Downloading {url}',
        total=int(data.info().get('Content-Length', -1)),
        unit='B',
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
    url = f'https://drive.usercontent.google.com/download?id={id}&confirm=t'
    return download_url(url, folder, filename)
