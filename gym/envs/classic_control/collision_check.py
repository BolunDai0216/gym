import math
import numpy as np
from shapely.geometry import Polygon
from shapely.geometry import Point


class Rectangle():
    def __init__(self, center, x, y, angle, offset=0):
        center = np.array(center) + np.array([offset*math.cos(angle), offset*math.sin(angle)])
        hlf_diag = math.sqrt(math.pow(x/2, 2) + math.pow(y/2, 2))
        beta = math.atan2(y, x)

        angle1 = math.pi - beta + angle
        angle2 = math.pi + beta + angle
        angle3 = -beta + angle
        angle4 = beta + angle

        self.point1 = center + np.array([hlf_diag*math.cos(angle1), hlf_diag*math.sin(angle1)])
        self.point2 = center + np.array([hlf_diag*math.cos(angle2), hlf_diag*math.sin(angle2)])
        self.point3 = center + np.array([hlf_diag*math.cos(angle3), hlf_diag*math.sin(angle3)])
        self.point4 = center + np.array([hlf_diag*math.cos(angle4), hlf_diag*math.sin(angle4)])

        self.polygon = Polygon([tuple(self.point1), tuple(self.point2),
                                tuple(self.point3), tuple(self.point4)])


class CollisionCheck():
    def __init__(self, rect1, rect2):
        self.rect1 = rect1
        self.rect2 = rect2

    def if_collide(self):
        return self.rect1.polygon.intersects(self.rect2.polygon)


class Contains():
    def __init__(self, x, y, rect):
        point = Point(x, y)
        self.point = point
        self.polygon = rect.polygon

    def contains(self):
        return self.polygon.contains(self.point)
