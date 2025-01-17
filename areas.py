from __future__ import annotations

import abc
import random

import numpy as np


class Area(abc.ABC):
    @abc.abstractmethod
    def contains_point(self, point: tuple[float, float] | np.ndarray) -> bool:
        """
        Determines whether a given point is located within the area defined by the
        respective class

        :param point: The coordinates of the point to check, specified as a tuple
            of floats (x, y) or an ndarray.
        :return: Whether the specified point is within the area, considering the
            allowable rect.
        """
        pass

    @property
    def name(self) -> str:
        pass


class Circle(Area):
    @property
    def name(self) -> str:
        return self._name

    def __init__(self, center_x: float, center_y: float, radius: float, filt_margin=0.3, name='Circle') -> None:
        self.center_x = center_x
        self.center_y = center_y
        self.radius = radius
        self._name = name
        self.filt_margin = filt_margin

    def contains_point(self, point: tuple[float, float] | np.ndarray) -> bool:
        if isinstance(point, np.ndarray):
            point = point.tolist()
        return (point[0] - self.center_x)**2 + (point[1] - self.center_y)**2 < (self.radius + self.filt_margin)**2

    def gen_points(self, n_points: int) -> list[tuple[float, float]]:
        point_list = list()

        for i in range(n_points):
            ang = 2 * np.pi * i / n_points
            x = self.center_x + self.radius * np.cos(ang)
            y = self.center_y + self.radius * np.sin(ang)
            point_list.append((x, y))

        return point_list


class ConvexPolygon(Area):
    @property
    def name(self) -> str:
        return self._name

    def __init__(self, points: list[tuple[float, float]], filt_margin=0.3, name='ConvexPolygon') -> None:
        """
        constructor for polygon class
        :type name: name for ConvexPolygon
        :param points: points in anticlockwise (!!!) order
        """
        self._name = name
        self.filt_margin = filt_margin
        self.points = np.array(points)

        self.hesse_eqs = list()

        rot_mat = np.array([[0., -1.], [1., 0.]])
        l = self.points.shape[0]
        for p_index in range(l):
            p1 = self.points[p_index - 1,:]
            p2 = self.points[p_index % l,:]

            # see https://de.wikipedia.org/wiki/Hessesche_Normalform
            v = np.reshape(p2 - p1, (2, 1))
            n = np.matmul(rot_mat, v)
            d = p1.dot(n)
            n = n / np.linalg.norm(n)
            inv = False
            if d < 0:
                n *= -1
                inv = True
            d = p1.dot(n)

            self.hesse_eqs.append((n, d, inv))

    def contains_point(self, point: tuple[float, float] | np.ndarray) -> bool:
        if isinstance(point, np.ndarray):
            point = point.reshape((1, 2))
        else:
            point = np.array(point).reshape((1, 2))

        # if any point lies behind a plane, it is on the outside of the polygon
        for n, d, inv in self.hesse_eqs:
            if (np.dot(point, n) < d + self.filt_margin) ^ inv:
                return False

        return True

    def gen_points(self, density: float, min_dist: float) -> list[tuple[float, float]]:
        """
        Place points randomly on the edge of a polygon
        :param density: points per len unit
        :param min_dist: minimum distance between two points
        :return: list of the created points (x, y)
        """

        point_list = list()

        n_pts = self.points.shape[0]
        for p_index in range(n_pts):
            p1 = self.points[p_index]
            p2 = self.points[(p_index + 1) % n_pts]
            dist = np.linalg.norm(p2 - p1)
            vec_norm = (p2 - p1) / dist

            # if max_dist >= min_dist, the random scattering is basically 0
            max_dist = 1 / density
            max_dist = max(max_dist, min_dist)

            point_list.append(p1.tolist())

            travelled = min_dist
            while travelled < dist - min_dist:
                # (dist - min_dist) because there will be the next edge of the polygon
                point = p1 + travelled * vec_norm
                point_list.append(point.tolist())
                travelled += random.uniform(min_dist, max_dist)

            # point_list.append(p1.tolist())
            # for _ in range(int(dist * density)):
            #     point = p1 + random.random() * vec_norm * dist
            #     point_list.append(point.tolist())

        return point_list


def gen_octaeda(
        center_x: float, center_y: float, w: float, h: float,
        rect_margin: float, filt_margin: float, name=''
) -> ConvexPolygon:
    edges = np.array([
        [center_x - w / 2, center_y + h / 2],
        [center_x - w / 2, center_y - h / 2],
        [center_x + w / 2, center_y - h / 2],
        [center_x + w / 2, center_y + h / 2],
    ])
    directions = np.array([(0, 1), (-1, 0), (0, -1), (1, 0)])
    points = list()

    l = len(edges)
    for i in range(l):
        p1 = edges[i-1] + directions[i-1] * rect_margin
        p2 = edges[i-1] + directions[i % l] * rect_margin
        points.append(p1)
        points.append(p2)

    if name:
        return ConvexPolygon(points, filt_margin, name=name)
    return ConvexPolygon(points, filt_margin)
