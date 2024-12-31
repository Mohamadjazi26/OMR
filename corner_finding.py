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
    rectangle = image_utils.find_squares(image)


    all_polygons: tp.List[
        geometry_utils.Polygon] = image_utils.find_polygons(
            rectangle[1],rectangles=rectangle[0], save_path=save_path)


    quadrilaterals: tp.List[geometry_utils.Polygon] = [
        poly for poly in all_polygons if len(poly) == 4
                              and geometry_utils.all_approx_square(poly)
                              and math_utils.all_approx_equal([abs(poly[0].x - poly[1].x), abs(poly[1].y - poly[2].y)],
                                                              tolerance=0.05)
    ]

    if save_path:
        image_utils.draw_polygons(rectangle[1], quadrilaterals, save_path / "all_quadrilaterals.jpg",labels=list(range(len(quadrilaterals))))

    top_left_squares = quadrilaterals[8] # 9 10 11
    top_right_squares = quadrilaterals[6] # 6 7 8
    bottom_right_squares = quadrilaterals[4] # 4 5
    bottom_left_squares = quadrilaterals[0] # 0 1 2 3

    top_left_corner =  geometry_utils.get_corner(
        top_left_squares, geometry_utils.Corner.TR)
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


