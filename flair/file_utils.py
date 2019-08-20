
from tqdm import tqdm as _tqdm
'\nUtilities for working with the local dataset cache. Copied from AllenNLP\n'
from pathlib import Path
from typing import Tuple, Optional, Sequence, cast
import os
import base64
import logging
import shutil
import tempfile
import re
from urllib.parse import urlparse
import mmap
import requests
import zipfile
import io
import flair
logger = logging.getLogger('flair')


def load_big_file(f):
    '\n    Workaround for loading a big pickle file. Files over 2GB cause pickle errors on certin Mac and Windows distributions.\n    :param f:\n    :return:\n    '
    logger.info(''.join(['loading file ', '{}'.format(f)]))
    with open(f, 'r+b') as f_in:
        bf = mmap.mmap(f_in.fileno(), 0)
        f_in.close()
    return bf


def url_to_filename(url: str, etag: str = None) -> str:
    "\n    Converts a url into a filename in a reversible way.\n    If `etag` is specified, add it on the end, separated by a period\n    (which necessarily won't appear in the base64-encoded filename).\n    Get rid of the quotes in the etag, since Windows doesn't like them.\n    "
    url_bytes = url.encode('utf-8')
    b64_bytes = base64.b64encode(url_bytes)
    decoded = b64_bytes.decode('utf-8')
    if etag:
        etag = etag.replace('"', '')
        return ''.join(['{}'.format(decoded), '.', '{}'.format(etag)])
    else:
        return decoded


def filename_to_url(filename: str) -> Tuple[(str, str)]:
    '\n    Recovers the the url from the encoded filename. Returns it and the ETag\n    (which may be ``None``)\n    '
    try:
        (decoded, etag) = filename.split('.', 1)
    except ValueError:
        (decoded, etag) = (filename, None)
    filename_bytes = decoded.encode('utf-8')
    url_bytes = base64.b64decode(filename_bytes)
    return (url_bytes.decode('utf-8'), etag)


def cached_path(url_or_filename: str, cache_dir: Path) -> Path:
    "\n    Given something that might be a URL (or might be a local path),\n    determine which. If it's a URL, download the file and cache it, and\n    return the path to the cached file. If it's already a local path,\n    make sure the file exists and then return the path.\n    "
    dataset_cache = (Path(flair.cache_root) / cache_dir)
    parsed = urlparse(url_or_filename)
    if (parsed.scheme in ('http', 'https')):
        return get_from_cache(url_or_filename, dataset_cache)
    elif ((parsed.scheme == '') and Path(url_or_filename).exists()):
        return Path(url_or_filename)
    elif (parsed.scheme == ''):
        raise FileNotFoundError('file {} not found'.format(url_or_filename))
    else:
        raise ValueError(
            'unable to parse {} as a URL or as a local path'.format(url_or_filename))


def unzip_file(file: Path, unzip_to: Path):
    from zipfile import ZipFile
    with ZipFile(file, 'r') as zipObj:
        zipObj.extractall(unzip_to)


def download_file(url: str, cache_dir: Path):
    cache_dir.mkdir(parents=True, exist_ok=True)
    filename = re.sub('.+/', '', url)
    cache_path = (cache_dir / filename)
    print(cache_path)
    (fd, temp_filename) = tempfile.mkstemp()
    logger.info('%s not found in cache, downloading to %s', url, temp_filename)
    req = requests.get(url, stream=True)
    content_length = req.headers.get('Content-Length')
    total = (int(content_length) if (content_length is not None) else None)
    progress = Tqdm.tqdm(unit='B', total=total)
    with open(temp_filename, 'wb') as temp_file:
        for chunk in req.iter_content(chunk_size=1024):
            if chunk:
                progress.update(len(chunk))
                temp_file.write(chunk)
    progress.close()
    logger.info('copying %s to cache at %s', temp_filename, cache_path)
    shutil.copyfile(temp_filename, str(cache_path))
    logger.info('removing temp file %s', temp_filename)
    os.close(fd)
    os.remove(temp_filename)
    progress.close()


def get_from_cache(url: str, cache_dir: Path = None) -> Path:
    "\n    Given a URL, look for the corresponding dataset in the local cache.\n    If it's not there, download it. Then return the path to the cached file.\n    "
    cache_dir.mkdir(parents=True, exist_ok=True)
    filename = re.sub('.+/', '', url)
    cache_path = (cache_dir / filename)
    if cache_path.exists():
        return cache_path
    response = requests.head(url, headers={
        'User-Agent': 'Flair',
    })
    if (response.status_code != 200):
        raise IOError(''.join(['HEAD request failed for url ', '{}'.format(
            url), ' with status code ', '{}'.format(response.status_code), '.']))
    if (not cache_path.exists()):
        (fd, temp_filename) = tempfile.mkstemp()
        logger.info('%s not found in cache, downloading to %s',
                    url, temp_filename)
        req = requests.get(url, stream=True, headers={
            'User-Agent': 'Flair',
        })
        content_length = req.headers.get('Content-Length')
        total = (int(content_length) if (content_length is not None) else None)
        progress = Tqdm.tqdm(unit='B', total=total)
        with open(temp_filename, 'wb') as temp_file:
            for chunk in req.iter_content(chunk_size=1024):
                if chunk:
                    progress.update(len(chunk))
                    temp_file.write(chunk)
        progress.close()
        logger.info('copying %s to cache at %s', temp_filename, cache_path)
        shutil.copyfile(temp_filename, str(cache_path))
        logger.info('removing temp file %s', temp_filename)
        os.close(fd)
        os.remove(temp_filename)
    return cache_path


def open_inside_zip(archive_path: str, cache_dir: Path, member_path: Optional[str] = None, encoding: str = 'utf8') -> iter:
    cached_archive_path = cached_path(archive_path, cache_dir=cache_dir)
    archive = zipfile.ZipFile(cached_archive_path, 'r')
    if (member_path is None):
        members_list = archive.namelist()
        member_path = get_the_only_file_in_the_archive(
            members_list, archive_path)
    member_path = cast(str, member_path)
    member_file = archive.open(member_path, 'r')
    return io.TextIOWrapper(member_file, encoding=encoding)


def get_the_only_file_in_the_archive(members_list: Sequence[str], archive_path: str) -> str:
    if (len(members_list) > 1):
        raise ValueError(('The archive %s contains multiple files, so you must select one of the files inside providing a uri of the type: %s' % (
            archive_path, format_embeddings_file_uri('path_or_url_to_archive', 'path_inside_archive'))))
    return members_list[0]


def format_embeddings_file_uri(main_file_path_or_url: str, path_inside_archive: Optional[str] = None) -> str:
    if path_inside_archive:
        return '({})#{}'.format(main_file_path_or_url, path_inside_archive)
    return main_file_path_or_url


class Tqdm():
    default_mininterval = 0.1

    @staticmethod
    def set_default_mininterval(value: float) -> None:
        Tqdm.default_mininterval = value

    @staticmethod
    def set_slower_interval(use_slower_interval: bool) -> None:
        "\n        If ``use_slower_interval`` is ``True``, we will dramatically slow down ``tqdm's`` default\n        output rate.  ``tqdm's`` default output rate is great for interactively watching progress,\n        but it is not great for log files.  You might want to set this if you are primarily going\n        to be looking at output through log files, not the terminal.\n        "
        if use_slower_interval:
            Tqdm.default_mininterval = 10.0
        else:
            Tqdm.default_mininterval = 0.1

    @staticmethod
    def tqdm(*args, **kwargs):
        new_kwargs = {
            'mininterval': Tqdm.default_mininterval,
            **kwargs,
        }
        return _tqdm(*args, **new_kwargs)
