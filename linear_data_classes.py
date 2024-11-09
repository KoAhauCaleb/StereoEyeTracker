import numpy as np


class Point:
    def __init__(self, x, y, z):
        pass


class Vector(Point):
    pass


class Line:
    def __init__(self, point, normal):
        pass


class Plane:
    def __init__(self, normal, scale):
        """
        Representation of equation [n1, n2, n3][x,y,z] = scale

        :param normal: vector normal to plane
        :param scale:
        """

        self.normal = normal
        self.scale = scale


class LineSeg:
    def __init__(self, x1, y1, z1, x2, y2, z2):
        """
        Line segment between two points, (x1, y1, z1) and (x2, y2, z2).
        Contains a midpoint property.

        :param x1:
        :param y1:
        :param z1:
        :param x2:
        :param y2:
        :param z2:
        """
        self.x1 = x1
        self.y1 = y1
        self.z1 = z1
        self.x2 = x2
        self.y2 = y2
        self.z2 = z2

    @property
    def point1(self):
        point1 = np.array([self.x1, self.y1, self.z1])
        return point1

    @property
    def point2(self):
        point2 = np.array([self.x2, self.y2, self.z2])
        return point2

    @property
    def midpoint(self):
        midpoint = ((self.x1 + self.x2) / 2, (self.y1 + self.y2) / 2, (self.z1 + self.z2) / 2)
        return midpoint
