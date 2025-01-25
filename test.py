from __future__ import annotations

import unittest

import numpy as np
from matplotlib import pyplot as plt

from areas import ConvexPolygon

SQUARE_PTS = [(-0.5, -0.5), (0.5, -0.5), (0.5, 0.5), (-0.5, 0.5)]
POINT_INSIDE_1 = (0.0, 0.0)
POINT_INSIDE_2 = (0.4, 0.4)
POINT_INSIDE_3 = (0.4, 0.0)
POINT_OUSIDE_1 = (0.6, 0.6)
POINT_OUSIDE_2 = (0.0, 0.6)


class Point2D:
    def __init__(self, tup: tuple[float, float]):
        x, y = tup
        self._point = np.array([[x], [y], [1.0]])

    def translate(self, tup) -> Point2D:
        x, y = tup
        translate_matrix = np.array([[1, 0, x],
                                      [0, 1, y],
                                      [0, 0, 1]])
        new_point = translate_matrix @ self._point
        return Point2D((float(new_point[0, 0]), float(new_point[1, 0])))

    def rotate(self, theta: float) -> Point2D:
        cos_theta = np.cos(np.radians(theta))
        sin_theta = np.sin(np.radians(theta))
        rotate_matrix = np.array([[cos_theta, -sin_theta, 0],
                                   [sin_theta, cos_theta, 0],
                                   [0, 0, 1]])
        new_point = rotate_matrix @ self._point
        return Point2D((float(new_point[0, 0]), float(new_point[1, 0])))

    def get_point(self) -> tuple[float, float]:
        return float(self._point[0, 0]), float(self._point[1, 0])


OCTAEDA_PTS = [Point2D((0.0, 0.0)).translate((1., 0.)).rotate(2 * np.pi * i / 8).get_point() for i in range(8)]


class PolygonTestCase:
    def __init__(self,
                 polygon_points: list[tuple[float, float]],
                 margin: float,
                 check_point: tuple):
        self.polygon = ConvexPolygon(polygon_points, margin)
        self.check_point = np.array(check_point)

    def contains_point(self) -> bool:
        return self.polygon.contains_point(self.check_point)


class PolygonTest(unittest.TestCase):
    def test1(self):
        points = SQUARE_PTS[:]
        polygon = ConvexPolygon(points, 0.)

        self.assertTrue(polygon.contains_point((0.0, 0.0)))

    def test2(self):
        # rotate square
        for i in range(4):
            # rotate point
            for j in range(4):
                # shift package in x dir
                for kx in range(-2, 3):
                    # shift package in y dir
                    for ky in range(-2, 3):
                        self.run_square_test(i, j, kx, ky)

    def run_square_test(self, i, j, kx, ky):
        points = SQUARE_PTS[:]
        points = [Point2D(p)
                  .rotate(2 * np.pi * i / 4)
                  .translate((float(kx), float(ky)))
                  .get_point() for p in points]
        polygon = ConvexPolygon(points, 0.)

        point_inside_1 = Point2D(POINT_INSIDE_1).rotate(2 * np.pi * i / 4).translate((float(kx), float(ky))).get_point()
        point_inside_2 = Point2D(POINT_INSIDE_2).rotate(2 * np.pi * i / 4).translate((float(kx), float(ky))).get_point()
        point_inside_3 = Point2D(POINT_INSIDE_3).rotate(2 * np.pi * i / 4).translate((float(kx), float(ky))).get_point()
        point_outside_1 = Point2D(POINT_OUSIDE_1).rotate(2 * np.pi * i / 4).translate(
            (float(kx), float(ky))).get_point()
        point_outside_2 = Point2D(POINT_OUSIDE_2).rotate(2 * np.pi * i / 4).translate(
            (float(kx), float(ky))).get_point()

        # if i == 0 and j == 0 and kx == -2 and ky == -1:
        #     inside_points = [point_inside_1, point_inside_2, point_inside_3]
        #     outside_points = [point_outside_1, point_outside_2]
        #     plt.plot([p[0] for p in points] + [points[0][0]], [p[1] for p in points] + [points[0][1]], '-',
        #              label='square')
        #     plt.plot([p[0] for p in inside_points], [p[1] for p in inside_points], marker='x', linestyle='',
        #              label='inside')
        #     plt.plot([p[0] for p in outside_points], [p[1] for p in outside_points], marker='x', linestyle='',
        #              label='outside')
        #     plt.legend()
        #     plt.show()

        self.assertTrue(polygon.contains_point(point_inside_1),
                        f'check point_inside_1, i=={i} and j=={j} and kx=={kx} and ky=={ky}')
        self.assertTrue(polygon.contains_point(point_inside_2),
                        f'check point_inside_1, i=={i} and j=={j} and kx=={kx} and ky=={ky}')
        self.assertTrue(polygon.contains_point(point_inside_3),
                        f'check point_inside_1, i=={i} and j=={j} and kx=={kx} and ky=={ky}')
        self.assertFalse(polygon.contains_point(point_outside_1),
                        f'check point_inside_1, i=={i} and j=={j} and kx=={kx} and ky=={ky}')
        self.assertFalse(polygon.contains_point(point_outside_2),
                        f'check point_inside_1, i=={i} and j=={j} and kx=={kx} and ky=={ky}')

    def test_detection(self):
        corner_points = [(0., 0.), (1., 0.), (1., 1.), (0., 1.)]
        poly = ConvexPolygon(corner_points, 0.)

        # test for point inside with margin 0
        point_in = (0.5, 0.5)
        self.assertTrue(poly.contains_point(point_in), f'{point_in} was detected as outside')

        # test for points outside with margin 0
        points_out = [
            (0.5, -0.5),
            (1.5,  0.5),
            (0.5,  1.5),
            (-0.5, 0.5),
        ]
        for point_out in points_out:
            self.assertFalse(poly.contains_point(point_out), f'{point_out} was detected as inside')

        # test for point inside with margin 0.51
        for point_out in points_out:
            self.assertTrue(
                poly.contains_point_with_margin(point_out, 0.51),
                f'{point_out} was detected as outside'
            )

        point_in = (0.1, 0.1)
        self.assertFalse(
            poly.contains_point_with_margin(point_in, -0.49),
            f'{point_in} was detected as inside, but is outside regarding margin'
        )




if __name__ == '__main__':
    unittest.main()
