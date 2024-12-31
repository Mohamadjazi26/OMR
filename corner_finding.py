import typing as tp
from time import sleep

import numpy as np
import geometry_utils
import image_utils
import list_utils
import math_utils
import pathlib

from math_utils import is_within_tolerance


class WrongShapeError(ValueError):
    pass


class CornerFindingError(RuntimeError):
    pass


def find_corner_marks(image: np.ndarray,
                      save_path: tp.Optional[pathlib.PurePath] = None
                      ) -> geometry_utils.Polygon:
    # rectangle = image_utils.find_squares(image)
    image = image_utils.dilate(image)
    hexagons: tp.List[geometry_utils.Polygon] = []
    quadrilaterals: tp.List[geometry_utils.Polygon] = []

    all_polygons: tp.List[
        geometry_utils.Polygon] = image_utils.find_polygons(
       image, save_path=save_path)

    for poly in all_polygons:
        if len(poly) == 6:
            hexagons.append(poly)
        elif len(poly) == 4 and math_utils.all_approx_equal([abs(poly[0].x - poly[1].x), abs(poly[1].y - poly[2].y)],tolerance=0.15):
            quadrilaterals.append(poly)
    print(len(quadrilaterals))

    if save_path:
        image_utils.draw_polygons(image, quadrilaterals, save_path / "all_quadrilaterals.jpg",labels=list(range(len(quadrilaterals))))


    img_height, img_width = image.shape[:2]
    top_left_squares, top_right_squares, bottom_left_squares, bottom_right_squares = None, None, None, None 
    for quad in quadrilaterals:
        centroid = geometry_utils.get_centroid(quad)
        if centroid.x < img_width / 2 and centroid.y < img_height / 2:
            top_left_squares = quad
        elif centroid.x >= img_width / 2 and centroid.y < img_height / 2:
            top_right_squares = quad
        elif centroid.x >= img_width / 2 and centroid.y >= img_height / 2:
            bottom_right_squares = quad
        elif centroid.x < img_width / 2 and centroid.y >= img_height / 2:
            bottom_left_squares = quad
    
    if not all([top_left_squares, top_right_squares, bottom_right_squares, bottom_left_squares]):
        raise CornerFindingError("Could not map the four corner quadrilaterals.")


    top_left_corner =  geometry_utils.get_corner(
        top_left_squares, geometry_utils.Corner.TL)
    top_right_corner = geometry_utils.get_corner(
        top_right_squares, geometry_utils.Corner.TR)
    bottom_right_corner = geometry_utils.get_corner(
        bottom_right_squares, geometry_utils.Corner.BR)
    bottom_left_corner = geometry_utils.get_corner(
        bottom_left_squares, geometry_utils.Corner.BL)


    grid_corners = [
    top_left_corner, top_right_corner,
    bottom_right_corner, bottom_left_corner
    ]

    if save_path:
        image_utils.draw_polygons(image, [grid_corners], save_path / "grid_limits.jpg")

    return grid_corners


