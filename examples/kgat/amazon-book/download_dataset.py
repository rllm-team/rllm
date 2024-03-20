import os
import requests
import zipfile


def download_file_from_github(url, file_path):
    if os.path.exists(file_path):
        print(f"{file_path} exists")
        return
    print(f"downloading to {file_path}")
    response = requests.get(url)
    with open(file_path, 'wb') as f:
        f.write(response.content)


def extract_zip(file_path, extract_dir):
    with zipfile.ZipFile(file_path, 'r') as zip_ref:
        zip_ref.extractall(extract_dir)


def download_dataset():
    # 要下载的文件的 URL 列表
    urls = [
        'https://github.com/LunaBlack/KGAT-pytorch/raw/' +
        'master/datasets/amazon-book/entity_list.txt',
        'https://github.com/LunaBlack/KGAT-pytorch/raw/' +
        'master/datasets/amazon-book/item_list.txt',
        'https://github.com/LunaBlack/KGAT-pytorch/raw/' +
        'master/datasets/amazon-book/kg_final.txt.zip',
        'https://github.com/LunaBlack/KGAT-pytorch/raw/' +
        'master/datasets/amazon-book/reaction_list.txt',
        'https://github.com/LunaBlack/KGAT-pytorch/raw/' +
        'master/datasets/amazon-book/test.txt',
        'https://github.com/LunaBlack/KGAT-pytorch/raw/' +
        'master/datasets/amazon-book/train.txt',
        'https://github.com/LunaBlack/KGAT-pytorch/raw/' +
        'master/datasets/amazon-book/train1.txt',
        'https://github.com/LunaBlack/KGAT-pytorch/raw/' +
        'master/datasets/amazon-book/user_list.txt',
        'https://github.com/LunaBlack/KGAT-pytorch/raw/' +
        'master/datasets/amazon-book/vaild1.txt'
    ]

    # 本地文件路径列表
    adr = os.getcwd()
    # print(adr)
    base_dir = adr + '/datasets/amazon-book'
    os.makedirs(base_dir, exist_ok=True)
    # print(base_dir)
    file_paths = [
        os.path.join(base_dir, 'entity_list.txt'),
        os.path.join(base_dir, 'item_list.txt'),
        os.path.join(base_dir, 'kg_final.txt.zip'),
        os.path.join(base_dir, 'reaction_list.txt'),
        os.path.join(base_dir, 'test.txt'),
        os.path.join(base_dir, 'train.txt'),
        os.path.join(base_dir, 'train1.txt'),
        os.path.join(base_dir, 'user_list.txt'),
        os.path.join(base_dir, 'vaild1.txt')
    ]

    # 下载文件
    for url, file_path in zip(urls, file_paths):
        download_file_from_github(url, file_path)

    # 解压'kg_final.txt.zip'文件并删除原始zip文件
    zip_file_path = os.path.join(base_dir, 'kg_final.txt.zip')
    extract_dir = base_dir
    # print(extract_dir)
    extract_zip(zip_file_path, extract_dir)
    os.remove(zip_file_path)
