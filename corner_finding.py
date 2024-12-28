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


class LMark:
    def __init__(self, polygon: geometry_utils.Polygon):

        if len(polygon) != 6:
            raise WrongShapeError("Incorrect number of points.")

        if not geometry_utils.all_approx_square(polygon):
            raise WrongShapeError("Corners are not square.")

        clockwise_polygon = geometry_utils.polygon_to_clockwise(polygon)
        side_lengths = geometry_utils.calc_side_lengths(clockwise_polygon)
        longest_sides_indexes = list_utils.find_greatest_value_indexes(
            side_lengths, n=2)

        if not list_utils.is_adjacent_indexes(side_lengths,
                                              *longest_sides_indexes):
            raise WrongShapeError("Longest sides are not adjacent.")

        # The longest sides should be about twice the length of the other sides
        unit_lengths = math_utils.divide_some(side_lengths,
                                              longest_sides_indexes, 2)
        if not math_utils.all_approx_equal(unit_lengths):
            raise WrongShapeError(
                "Longest sides are not twice the length of the other sides.")

        self.polygon = list_utils.arrange_index_to_first(
            clockwise_polygon,
            list_utils.determine_which_is_next(polygon,
                                               *longest_sides_indexes))
        self.unit_length = math_utils.mean(unit_lengths)


class SquareMark:
    def __init__(self,
                 polygon: geometry_utils.Polygon,
                 target_size: tp.Optional[float] = None):

        if len(polygon) != 4:
            raise WrongShapeError("Incorrect number of points.")

        if not geometry_utils.all_approx_square(polygon):
            raise WrongShapeError("Corners are not square.")

        side_lengths = geometry_utils.calc_side_lengths(polygon)
        if not math_utils.all_approx_equal(side_lengths, target_size):
            raise WrongShapeError(
                "Side lengths are not equal or too far from target_size.")

        clockwise = geometry_utils.polygon_to_clockwise(polygon)
        if clockwise[0] is polygon[0]:
            self.polygon = clockwise
        else:
            self.polygon = list_utils.arrange_index_to_first(
                clockwise,
                len(clockwise) - 1)
        self.unit_length = math_utils.mean(side_lengths)

    def __eq__(self, other: 'SquareMark') -> bool:

        return math_utils.all_approx_equal(self.unit_length, other.unit_length)

#---------------------------------------------------------------
# def find_corner_marks(image: np.ndarray, save_path: tp.Optional[pathlib.PurePath] = None) -> geometry_utils.Polygon:
#     """Detect the four corner squares of the grid."""
#     # all_polygons = image_utils.find_polygons(image, save_path=save_path)
#     all_polygons: tp.List[
#         geometry_utils.Polygon] = image_utils.find_polygons(
#         image, save_path=save_path)
#
#     quadrilaterals = [poly for poly in all_polygons if len(poly) == 4]
#
#     hexagons: tp.List[geometry_utils.Polygon] = []
#     # quadrilaterals: tp.List[geometry_utils.Polygon] = []
#
#     # for poly in all_polygons:
#     #     if len(poly) == 6:
#     #         hexagons.append(poly)
#     #     elif len(poly) == 4:
#     #         quadrilaterals.append(poly)
#     for i in range(len(hexagons)):
#         hexagon = hexagons[i]
#         try:
#             l_mark = LMark(hexagon)
#         except WrongShapeError:
#             continue
#
#         # To construct the basis, we use points 0, 4, 5 of the L. The points are in CW order from
#         # the top-left corner, so these are the top-left, bottom-left, and bottom-right-of-bottom-side
#         # points.
#
#         basis_transformer = geometry_utils.ChangeOfBasisTransformer(
#             l_mark.polygon[0], l_mark.polygon[5], l_mark.polygon[4])
#         nominal_to_right_side = 50 - 0.5
#         # Since the side of the L is twice as long as it is wide, divide y by two as the new-basis
#         # system will be squashed
#         nominal_to_bottom = ((64 - 0.5) / 2)
#         # We can afford to allow a decently large error margin here since we are just searching for
#         # reference points and the rough coordinate system established based on the L is very
#         # sensitive to noise.
#         x_tolerance = 0.2 * nominal_to_right_side
#         y_tolerance = 0.2 * nominal_to_bottom
#
#         if save_path:
#             # Purely for diagnostic output - save the grid tolerance boxes to a file. This is
#             # complicated, but useful for debugging and only is enabled when requested.
#             # Nominal polygon of edges
#             nominal_poly_new_basis = [
#                 geometry_utils.Point(0.0, 0.0),
#                 geometry_utils.Point(nominal_to_right_side, 0.0),
#                 geometry_utils.Point(nominal_to_right_side, nominal_to_bottom),
#                 geometry_utils.Point(0.0, nominal_to_bottom)
#             ]
#             # Boxes within which corner centroids can be found
#             corner_tolerance_polys_new_basis = [
#                 [
#                     geometry_utils.Point(x + x_tolerance, y - y_tolerance),
#                     geometry_utils.Point(x + x_tolerance, y + y_tolerance),
#                     geometry_utils.Point(x - x_tolerance, y + y_tolerance),
#                     geometry_utils.Point(x - x_tolerance, y - y_tolerance)
#                 ] for [x, y] in [
#                     [nominal_to_right_side, 0.5],
#                     [nominal_to_right_side, nominal_to_bottom],
#                     [0.5, nominal_to_bottom]
#                 ]
#             ]
#             polys = [basis_transformer.poly_from_basis(nominal_poly_new_basis), hexagon] + [
#                 basis_transformer.poly_from_basis(p) for p in corner_tolerance_polys_new_basis
#             ]
#             image_utils.draw_polygons(
#                 image,
#                 polys,
#                 save_path / f"grid_corner_tolerances_{i}.png",
#                 thickness=2
#             )
#--------------------------------------------------------------------------------------------------
    # top_left = None
    # top_right = None
    # bottom_left = None
    # bottom_right = None
    #
    # for quadrilateral in quadrilaterals:
    #     centroid = geometry_utils.guess_centroid(quadrilateral)
    #     if is_within_tolerance(centroid.x, 0.5, 0.2) and is_within_tolerance(centroid.y, 0.5, 0.2):
    #         top_left = quadrilateral
    #     elif is_within_tolerance(centroid.x, 0.9, 0.2) and is_within_tolerance(centroid.y, 0.5, 0.2):
    #         top_right = quadrilateral
    #     elif is_within_tolerance(centroid.x, 0.5, 0.2) and is_within_tolerance(centroid.y, 0.9, 0.2):
    #         bottom_left = quadrilateral
    #     elif is_within_tolerance(centroid.x, 0.9, 0.2) and is_within_tolerance(centroid.y, 0.9, 0.2):
    #         bottom_right = quadrilateral
    #
    # if top_left and top_right and bottom_left and bottom_right:
    #     grid_corners = [top_left[0], top_right[0], bottom_right[0], bottom_left[0]]
    #     if save_path:
    #         image_utils.draw_polygons(image, [grid_corners], save_path / "grid_limits.jpg")
    #     return grid_corners
    # else:
    #     "continue"
    #     #raise RuntimeError("Could not find all four grid corners.")

#----------------------------------------------------------------

def find_corner_marks(image: np.ndarray,
                      save_path: tp.Optional[pathlib.PurePath] = None
                      ) -> geometry_utils.Polygon:

    all_polygons: tp.List[
        geometry_utils.Polygon] = image_utils.find_polygons(
            image, save_path=save_path)

    quadrilaterals: tp.List[geometry_utils.Polygon] = [
        poly for poly in all_polygons if len(poly) == 4 and geometry_utils.all_approx_square(poly)
    ]
    
    if save_path:
        image_utils.draw_polygons(image, quadrilaterals, save_path / "all_quadrilaterals.jpg")


    top_squares = []
    bottom_squares = []


    centroids = [
        (geometry_utils.guess_centroid(poly), poly)
        for poly in quadrilaterals
    ]

    sorted_centroids = sorted(centroids, key=lambda c: c[0].y)

    # Group top and bottom squares based on their y-coordinates
    top_squares = sorted_centroids[:2]
    bottom_squares = sorted_centroids[2:]
    print(len(top_squares))
    print(len(bottom_squares))
    if len(top_squares) != 2 or len(bottom_squares) != 2:
        raise CornerFindingError("Could not find the required four squares.")

    # Ensure squares are ordered by x-coordinate within each group
    top_squares = sorted(top_squares, key=lambda c: c[0].x)
    bottom_squares = sorted(bottom_squares, key=lambda c: c[0].x)

    grid_corners = [
        top_squares[0][1][0],  # Top-left
        top_squares[1][1][0],  # Top-right
        bottom_squares[1][1][0],  # Bottom-right
        bottom_squares[0][1][0]  # Bottom-left
    ]
    print(len(grid_corners))
    sleep(20)
    # top_left_corner = l_mark.polygon[0]
    # top_right_corner = geometry_utils.get_corner_wrt_basis(
    #     top_right_squares[0].polygon, geometry_utils.Corner.TR, basis_transformer)
    # bottom_right_corner = geometry_utils.get_corner_wrt_basis(
    #     bottom_right_squares[0].polygon, geometry_utils.Corner.BR, basis_transformer)
    # bottom_left_corner = geometry_utils.get_corner_wrt_basis(
    #     bottom_left_squares[0].polygon, geometry_utils.Corner.BL, basis_transformer)

    # grid_corners = [
    #     top_left_corner, top_right_corner,
    #     bottom_right_corner, bottom_left_corner
    # ]

    # Corners [top-left, top-right, bottom-right, bottom-left]
    if save_path:
        image_utils.draw_polygons(image, [grid_corners], save_path / "grid_limits.jpg")

    return grid_corners


