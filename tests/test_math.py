import unittest
from tracker_math import TrackerMath

class TrackerMathTest(unittest.TestCase):
    def setUp(self):
        self.math = TrackerMath()

    def test_plane_intersection_on_yz_plane(self):
        line = (0, 0, 0, 1, 1, 1)
        plane = (1, 0, 0, 0)
        result = self.math.plane_intersection(line, plane)
        self.assertTupleEqual(result, (0, 0, 0))

    def test_plane_intersection_on_xyz_plane(self):
        line = (0, 0, 0, 1, 1, 1)
        plane = (1, 1, 1, 0)
        result = self.math.plane_intersection(line, plane)
        self.assertTupleEqual(result, (0, 0, 0))

    def test_plane_intersection_not_at_origin(self):
        line = (1, 1, 1, 1, 0, 0)
        plane = (1, 0, 0, 1)
        result = self.math.plane_intersection(line, plane)
        self.assertTupleEqual(result, (1, 1, 1))

    def test_plane_intersection(self):
        line = (2, 1, 0, -1, 1, 3)
        plane = (3, -2, 1, 10)
        result = self.math.plane_intersection(line, plane)
        self.assertTupleEqual(result, (5, -2, -9))

    def test_line_from_2_points(self):
        point1 = (1, 2, 3)
        point2 = (4, 5, 6)
        expected_result = (
        1, 2, 3, 3, 3, 3)  # line equation parameters formed by subtracting point1 coordinates from point2
        self.assertEqual(self.math.line_from_2_points(point1, point2), expected_result)

    def test_line_from_2_points_same_points(self):
        point1 = point2 = (1, 2, 3)
        expected_result = None  # as points are same, no line exists
        self.assertEqual(self.math.line_from_2_points(point1, point2), expected_result)

    def test_line_from_2_points_negative_coordinates(self):
        point1 = (-1, -2, -3)
        point2 = (4, 5, 6)
        expected_result = (-1, -2, -3, 5, 7, 9)  # line equation parameters formed by subtracting point1 coordinates from point2
        self.assertEqual(self.math.line_from_2_points(point1, point2), expected_result)

    def test_short_line_seg_01(self):
        line1 = self.math.line_from_2_points((1, 0, 0), (1, 1, 1))
        line2 = self.math.line_from_2_points((0, 0, 0), (0, 0, 1))
        d, t, q1, q2 = self.math.shortest_line_segment(line1, line2)
        self.assertTupleEqual(t, (0, 0, 1))

    def test_short_line_seg_02(self):
        line1 = (0, 1, 3, 1, 4, 2)
        line2 = (1, 1, 3, 0, 2, 4)
        dist, t, q1, q2 = self.math.shortest_line_segment(line1, line2)
        print(f"{t=}\n {q1=}\n {q2=}")
        self.assertAlmostEqual(dist, 0.94, places=2)

    def test_short_line_seg_02(self):
        line1 = (1, 1, 0, 2, -1, 1)
        line2 = (2, 1, -1, 3, -5, 2)
        dist, t, q1, q2 = self.math.shortest_line_segment(line1, line2)
        self.assertAlmostEqual(dist, 1.30, places=2)

    def test_short_line_seg_03(self):
        line1 = (0, 3, 32, 12, 423, 23)
        line2 = (13, 14, 34, 3, 254, 43)
        dist, t, q1, q2 = self.math.shortest_line_segment(line1, line2)
        self.assertAlmostEqual(dist, 12.75, places=2)

    def test_short_line_seg_04(self):
        line1 = (0, 1, 2, 2, 3, 4)
        line2 = (0, 1, 2, 2, 3, 4)
        dist, t, q1, q2 = self.math.shortest_line_segment(line1, line2)
        self.assertAlmostEqual(dist, 0, places=2)


    def test_short_line_seg_05(self):
        line1 = (0, 1, 2, 2, 3, 4)
        line2 = (0, 1, 2, 4, 6, 8)
        dist, t, q1, q2 = self.math.shortest_line_segment(line1, line2)
        self.assertAlmostEqual(dist, 0, places=2)

    def test_short_line_seg_06(self):
        line1 = (0, 1, 2, 2, 3, 4)
        line2 = (0, 1, 2, 2, 5, 4)
        dist, t, q1, q2 = self.math.shortest_line_segment(line1, line2)
        self.assertAlmostEqual(dist, 0, places=2)


if __name__ == '__main__':
    unittest.main()