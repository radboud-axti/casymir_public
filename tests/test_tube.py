import unittest
from casymir.casymir import Tube


class TestTube(unittest.TestCase):

    def test_initialization(self):
        # Test the default initialization of the Tube object
        tube = Tube()

        self.assertEqual(tube.target_angle, 25)
        self.assertEqual(tube.target, 'W')
        self.assertEqual(tube.SID, 65)
        self.assertEqual(tube.filter, [])
        self.assertEqual(tube.external_filter, [])

    def test_fill_from_dict(self):
        # Test initializing the Tube object with a dictionary
        tube_data = {
            'target_angle': 10,
            'target': 'Mo',
            'SID': 90,
            'filter': [('Be', 0.5)],
            'external_filter': [('Al', 1.0)]
        }

        tube = Tube(tube_data)
        self.assertEqual(tube.target_angle, 10)
        self.assertEqual(tube.target, 'Mo')
        self.assertEqual(tube.SID, 90)
        self.assertEqual(tube.filter, [('Be', 0.5)])
        self.assertEqual(tube.external_filter, [('Al', 1.0)])


if __name__ == '__main__':
    unittest.main()
