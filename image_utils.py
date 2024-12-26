import pathlib
import typing as tp
import cv2
import numpy as np
from numpy import ma
import geometry_utils
import argparse
from pathlib import Path

SUPPORTED_IMAGE_EXTENSIONS = [".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif"]

def convert_to_grayscale(image: np.ndarray,
                         save_path: tp.Optional[pathlib.PurePath] = None
                         ) -> np.ndarray:
    result = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    if save_path:
        save_image(save_path / "grayscale.jpg", result)
    return result


def remove_hf_noise(image: np.ndarray,
                    save_path: tp.Optional[pathlib.PurePath] = None
                    ) -> np.ndarray:

    sigma = min(get_dimensions(image)) * (5.6569e-4)
    result = cv2.GaussianBlur(image, (0, 0), sigmaX=sigma)
    if save_path:
        save_image(save_path / "noise_filtered.jpg", result)
    return result


def detect_edges(image: np.ndarray,
                 save_path: tp.Optional[pathlib.PurePath] = None
                 ) -> np.ndarray:
    low_threshold = 100
    result = cv2.Canny(image,
                       low_threshold,
                       low_threshold * 3,
                       L2gradient=True,
                       edges=3)
    if save_path:
        save_image(save_path / "edges.jpg", result)
    return result


def find_contours(edges: np.ndarray) -> np.ndarray:
    contours, _ = cv2.findContours(edges, cv2.RETR_TREE,
                                   cv2.CHAIN_APPROX_SIMPLE)
    return contours


def get_image(path: pathlib.PurePath,
              save_path: tp.Optional[pathlib.PurePath] = None) -> np.ndarray:
    result = cv2.imread(str(path))
    if save_path:
        save_image(save_path / "original.jpg", result)
    return result


def save_image(path: pathlib.PurePath, image: np.ndarray):
    cv2.imwrite(str(path), image)


def find_polygons(image: np.ndarray,
                  save_path: tp.Optional[pathlib.PurePath] = None
                  ) -> tp.List[geometry_utils.Polygon]:
    edges = detect_edges(image, save_path=save_path)
    all_contours = find_contours(edges)
    polygons = [
        geometry_utils.approx_poly(contour) for contour in all_contours
    ]
    return polygons


def get_dimensions(image: np.ndarray) -> tp.Tuple[int, int]:
    return image.shape[0], image.shape[1]


def threshold(image: np.ndarray,
              save_path: tp.Optional[pathlib.PurePath] = None) -> np.ndarray:
    gray_image = convert_to_grayscale(image)
    _, result = cv2.threshold(gray_image, 0, 255,
                              cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    if save_path:
        save_image(save_path / "thresholded.jpg", result)
    return result


def prepare_scan_for_processing(image: np.ndarray,
                                save_path: tp.Optional[pathlib.PurePath] = None
                                ) -> np.ndarray:

    without_noise = remove_hf_noise(image, save_path=save_path)
    result = threshold(without_noise, save_path=save_path)
    return result


def get_fill_percent(matrix: tp.Union[np.ndarray, ma.MaskedArray]) -> float:
    try:
        return 1 - (matrix.mean() / 255)
    except ZeroDivisionError:
        return 0


def dilate(image: np.ndarray,
           save_path: tp.Optional[pathlib.PurePath] = None) -> np.ndarray:

    result = cv2.dilate(image, np.ones((3, 3), np.uint8), iterations=1)
    if save_path:
        save_image(save_path / "dilated.jpg", result)
    return result


def bw_to_bgr(image: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(image.copy(), cv2.COLOR_GRAY2BGR)


def draw_polygons(image: np.ndarray,
                  polygons: tp.List[geometry_utils.Polygon],
                  full_save_path: tp.Optional[pathlib.PurePath] = None,
                  thickness: int = 1) -> np.ndarray:
    points = [np.array([[p.x, p.y] for p in poly], np.int32).reshape((-1, 1, 2)) for poly in polygons]
    result = cv2.polylines(bw_to_bgr(image), points, True, (0, 0, 255), thickness)
    if full_save_path:
        save_image(full_save_path, result)
    return result

