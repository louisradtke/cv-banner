import unittest

from areas import ConvexPolygon


class PolygonTest(unittest.TestCase):
    def test_detection(self):
        poly = ConvexPolygon([(0., 0.), (1., 0.), (1., 1.), (0., 1.)], 0.)

        point_in = (0.5, 0.5)
        self.assertTrue(poly.contains_point(point_in), f'{point_in} was detected as outside')

        points_out = [
            (0.5, -0.5),
            (1.5,  0.5),
            (0.5,  1.5),
            (-0.5, 0.5),
        ]
        for point_out in points_out:
            self.assertFalse(poly.contains_point(point_out), f'{point_out} was detected as inside')


if __name__ == '__main__':
    unittest.main()
