"""General geometry-related math utilities."""

import enum
import math
import typing as tp

import cv2
import numpy as np

import list_utils
import math_utils


class Point:
    x: float
    y: float
    """Represents a point on a 2d plane in x, y form."""

    def __init__(self, x: float, y: float):
        """Create a new Point."""
        self.x = x
        self.y = y


Polygon = tp.List[Point]


class Line:
    slope: float
    point: Point

    def __init__(self, slope: float, point: Point):
        self.point = point
        self.slope = slope

    def __call__(self, x: float) -> float:
        return (self.slope * (x - self.point.x)) + self.point.y


def contour_to_polygon(contour: np.ndarray) -> Polygon:
    return [Point(vertex[0][0], vertex[0][1]) for vertex in contour]


def polygon_to_contour(polygon: Polygon) -> np.ndarray:
    return np.array([[[point.x, point.y]] for point in polygon])


def approx_poly(contour: np.ndarray) -> Polygon:
    perimeter = cv2.arcLength(contour, True)
    simple = cv2.approxPolyDP(contour, 0.05 * perimeter, True)
    polygon = contour_to_polygon(simple)
    return polygon_to_clockwise(polygon)


def polygon_to_clockwise(polygon: Polygon) -> Polygon:
    clockwise = cv2.contourArea(polygon_to_contour(polygon), True) >= 0
    if clockwise:
        return polygon
    else:
        return list(reversed(polygon))


def calc_2d_dist(point_a: Point, point_b: Point) -> float:
    return math.sqrt((point_a.x - point_b.x) ** 2 + (point_a.y - point_b.y) ** 2)


def calc_angle(end_a: Point, shared: Point, end_b: Point) -> float:
    mag_a = calc_2d_dist(shared, end_a)
    mag_b = calc_2d_dist(shared, end_b)
    dist_ab = calc_2d_dist(end_a, end_b)
    cosine = (mag_a ** 2 + mag_b ** 2 - dist_ab ** 2) / (2 * mag_a * mag_b)
    angle = abs(math.acos(round(cosine, 4)))
    return angle if angle <= 180 else angle - 180


def calc_corner_angles(contour: Polygon) -> tp.List[float]:
    result = []
    for i, point in enumerate(contour):
        previous_point = contour[list_utils.prev_index(contour, i)]
        next_point = contour[list_utils.next_index(contour, i)]
        result.append(calc_angle(previous_point, point, next_point))
    return result


def calc_side_lengths(contour: Polygon) -> tp.List[float]:
    result = []
    for i, point in enumerate(contour):
        next_point = contour[list_utils.next_index(contour, i)]
        result.append(calc_2d_dist(point, next_point))
    return result

def all_approx_square(contour: Polygon) -> bool:
    angles = calc_corner_angles(contour)
    return math_utils.all_approx_equal(angles, math.pi / 2)


def line_from_points(point_a: Point, point_b: Point) -> Line:
    run = (point_a.x - point_b.x)
    slope = (point_a.y - point_b.y) / run if run != 0 else math.inf
    return Line(slope, point_a)


def get_perpendicular_line(line: Line,
                           point: tp.Optional[Point] = None) -> Line:
    return rotate_line(line, math.pi / 2, point)


def rotate_line(line: Line, theta: float,
                point: tp.Optional[Point] = None) -> Line:
    new_angle = math.atan(line.slope) + theta
    new_slope = math.tan(new_angle)
    return Line(new_slope, point if point is not None else line.point)


def calc_angle_between(line_a: Line, line_b: Line) -> float:
    angle_a = math.atan(line_a.slope)
    angle_b = math.atan(line_b.slope)
    return abs(angle_a - angle_b)


InequalityLine = tp.Tuple[Line, math_utils.InequalityTypes]

def is_in_inequalities(point: Point,
                       inequalities: tp.List[InequalityLine]) -> bool:
    for inequality in inequalities:
        compare_value = inequality[0](point.x)

        if ((inequality[1] is math_utils.InequalityTypes.GT
             and point.y <= compare_value)
                or (inequality[1] is math_utils.InequalityTypes.GTE
                    and point.y < compare_value)
                or (inequality[1] is math_utils.InequalityTypes.LT
                    and point.y >= compare_value)
                or (inequality[1] is math_utils.InequalityTypes.LTE
                    and point.y > compare_value)
                or (inequality[1] is math_utils.InequalityTypes.NE
                    and point.y == compare_value)):
            return False
    return True


def create_range_check_fn(*inequalities: tp.List[InequalityLine]
                          ) -> tp.Callable[[Point], bool]:
    return lambda point: is_in_inequalities(point, inequalities)


def offset_line(line: Line, offset_point: Point) -> Line:
    return Line(line.slope, offset_point)


def extend_ray(a: Point, b: Point, distance: float):
    theta = math.atan2((b.y - a.y), (b.x - a.x))
    dx = math.cos(theta) * distance
    dy = math.sin(theta) * distance
    return Point(b.x + dx, b.y + dy)


class ChangeOfBasisTransformer:
    def __init__(self, origin: Point, bottom_left: Point, bottom_right: Point):
        target_origin = Point(0, 0)
        target_bl = Point(0, 1)
        target_br = Point(1, 1)
        target_matrix = np.array([[target_origin.x], [target_bl.x], [target_br.x],
                                  [target_origin.y], [target_bl.y], [target_br.y]],
                                 float)

        from_matrix = np.array([[origin.x, origin.y, 1, 0, 0, 0],
                                [bottom_left.x, bottom_left.y, 1, 0, 0, 0],
                                [bottom_right.x, bottom_right.y, 1, 0, 0, 0],
                                [0, 0, 0, origin.x, origin.y, 1],
                                [0, 0, 0, bottom_left.x, bottom_left.y, 1],
                                [0, 0, 0, bottom_right.x, bottom_right.y, 1]],
                               float)

        result = np.matmul(np.linalg.inv(from_matrix), target_matrix)
        self._transformation_matrix = np.array([[result[0][0], result[1][0]],
                                                [result[3][0], result[4][0]]])
        self._transformation_matrix_inv = np.linalg.inv(self._transformation_matrix)
        self._rotation_matrix = np.array([[result[2][0]], [result[5][0]]])

    def to_basis(self, point: Point) -> Point:
        point_vector = np.array([[point.x], [point.y]], float)
        result = np.matmul(self._transformation_matrix,
                           point_vector) + self._rotation_matrix
        return Point(result[0][0], result[1][0])

    def from_basis(self, point: Point) -> Point:
        point_vector = np.array([[point.x], [point.y]], float)
        result = np.matmul(self._transformation_matrix_inv,
                           (point_vector - self._rotation_matrix))
        return Point(result[0][0], result[1][0])

    def poly_to_basis(self, polygon: Polygon) -> Polygon:
        return [self.to_basis(point) for point in polygon]

    def poly_from_basis(self, polygon: Polygon) -> Polygon:
        return [self.from_basis(point) for point in polygon]


def guess_centroid(quadrilateral: Polygon) -> Point:
    """Guesses an approximate centroid. Works well for squares."""
    xs = [p.x for p in quadrilateral]
    ys = [p.y for p in quadrilateral]
    return Point(math_utils.mean([max(xs), min(xs)]),
                 math_utils.mean([max(ys), min(ys)]))

def get_centroid(polygon: Polygon) -> Point:
    """Calculate the centroid of a polygon."""
    xs = [point.x for point in polygon]
    ys = [point.y for point in polygon]
    return Point(sum(xs) / len(xs), sum(ys) / len(ys))

class Corner(enum.Enum):
    TL = (1, 0)
    TR = (1, 1)
    BR = (0, 1)
    BL = (0, 0)


class Orientation(enum.Enum):
    VERTICAL = enum.auto()
    HORIZONTAL = enum.auto()


def get_corner(square: Polygon, corner: Corner) -> Point:
    xs = [p.x for p in square]
    highest_xs = sorted(list_utils.find_greatest_value_indexes(xs, 2))
    side_points = [
        p for i, p in enumerate(square)
        if (corner.value[1] == 1 and i in highest_xs) or (
                corner.value[1] == 0 and i not in highest_xs)
    ]
    side_ys = [p.y for p in side_points]
    [highest_y] = list_utils.find_greatest_value_indexes(side_ys, 1)
    corner_point = side_points[highest_y] if (
            corner.value[0] == 0) else side_points[list_utils.next_index(
        side_points, highest_y)]
    return corner_point


def get_corner_wrt_basis(square: Polygon, corner: Corner, basis: ChangeOfBasisTransformer) -> Point:
    transformed_square = basis.poly_to_basis(square)
    transformed_corner = get_corner(transformed_square, corner)
    return basis.from_basis(transformed_corner)


def crop_rectangle(top_left_corner: Point, bottom_right_corner: Point,
                   crop_fraction: float) -> tp.Tuple[Point, Point]:
    height = abs(top_left_corner.y - bottom_right_corner.y)
    dy = crop_fraction * height / 2
    width = abs(top_left_corner.x - bottom_right_corner.x)
    dx = crop_fraction * width / 2
    return Point(top_left_corner.x + dx,
                 top_left_corner.y + dy), Point(bottom_right_corner.x - dx,
                                                bottom_right_corner.y - dy)