"""Functions and utilities related to importing and exporting files."""

import pathlib
import typing as tp

from image_utils import SUPPORTED_IMAGE_EXTENSIONS
from str_utils import strip_double_quotes


def list_file_paths(directory: pathlib.Path) -> tp.List[pathlib.Path]:
    return [item for item in directory.iterdir() if item.is_file()]


def filter_by_extensions(files: tp.Sequence[pathlib.Path],
                         extensions: tp.List[str]) -> tp.List[pathlib.Path]:
    return [file for file in files if "".join(file.suffixes) in extensions]

def filter_images(files: tp.Sequence[pathlib.Path]) -> tp.List[pathlib.Path]:
    return filter_by_extensions(files, SUPPORTED_IMAGE_EXTENSIONS)

def parse_path_arg(path_arg: str) -> pathlib.Path:
    return pathlib.Path(strip_double_quotes(path_arg))
