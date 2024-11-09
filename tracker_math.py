import numpy as np
from numpy.linalg import solve, norm


class TrackerMath:
    def __init__(self):
        self.debug = False

    def line_from_2_points(self, point1, point2):
        """
        Returns parameters of a line intersecting 2 points.

        Will return None if points both points are equal

        :param point1: tuple (x, y, z)
        :param point2: tuple (x, y, z)
        :return: line: tuple (x1, y1, z1, a, b, c) forming the equations x = x1 + a * , y = y1 + b * t, z = z1 + c * t
        """
        if point1 == point2:
            return None

        x1, y1, z1 = point1
        x2, y2, z2 = point2

        # Define parallel vector by subtracting point1 from point2
        a = x2 - x1
        b = y2 - y1
        c = z2 - z1

        line = (x1, y1, z1, a, b, c)

        return line

    def plane_intersection(self, line, plane):
        """
        Finds the point where a line intersects a plane.

        :param line: tuple (x1, y1, z1, a, b, c) forming the equations x = x1 + a * , y = y1 + b * t, z = z1 + c * t
        :param plane: tuple (m, n, o, p) forming the equation (m * x) + (n * y) + (0 * z) = p
        :return: point: tuple (x, y, z)
        """

        x1, y1, z1 = line[0:3]
        a, b, c = line[3:6]
        m, n, o, p = plane

        # Find t
        q = (m * x1) + (n * y1) + (o * z1)
        r = (m * a) + (n * b) + (o * c)
        t = (p - q) / r

        # substitute out t in original line equations
        x = x1 + t * a
        y = y1 + t * b
        z = z1 + t * c

        # Create a point out of the individual line equations results
        intersection = (x, y, z)

        return intersection

    def shortest_line_segment(self, line1, line2):
        """
        Finds the 2 endpoints of the shortest line segment between two lines.

        :param line1: tuple (x1, y1, z1, a, b, c) forming the equations x = x1 + a * t, y = y1 + b * t, z = z1 + c * t
        :param line2: tuple (x1, y1, z1, a, b, c) forming the equations x = x1 + a * t, y = y1 + b * t, z = z1 + c * t
        :return: line_seg: tuple (x1, y1, z1, x2, y2, z2)

        source: https://math.stackexchange.com/questions/1993953/closest-points-between-two-lines
        """

        point1 = np.array(line1[0:3])
        vec1 = np.array(line1[3:6])
        point2 = np.array(line2[0:3])
        vec2 = np.array(line2[3:6])

        # ensure lines are not equal, return distance of 0 and 2 equivalent points otherwise
        # not worried about floats unless they are exactly equal
        # TODO: Make it work when lines are equal in cases where slope is common ratio, and intersection points are
        #  different
        if line1 == line2:
            # Return a distance of 0 if lines are equal
            return 0, (0, 0, 0), point1, point1

        # find the slope perpendicular to both lines by taking the cross product of the 2 original slopes
        vec3 = np.cross(vec2, vec1)

        if self.debug:
            print(f"{vec3=}")

        # solve for point1 + t1 * vec1 + point3 + t3 * vec3 = point2 + t2 * vec2

        # simplify to [v1, -v2, v3][t1, t2, t3] = p2 - p1
        RHS = point2 - point1
        LHS = np.array([vec1, -vec2, vec3]).T

        # solve for [t1, t2, t3]
        t = solve(LHS, RHS)

        if self.debug:
            print(f"{vec3=}")
            print(f"{RHS=}")
            print(f"{LHS=}")
            print(f"{t=}")

        q1 = point1 + t[0] * vec1
        q2 = point2 + t[1] * vec2
        dist = float(abs(norm(vec3) * t[2]))

        if self.debug:
            print(f"{dist=}")
        return dist, tuple(t.tolist()), tuple(q1.tolist()), tuple(q2.tolist())




