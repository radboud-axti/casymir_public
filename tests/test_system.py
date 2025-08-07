import unittest
from casymir.casymir import System
import yaml
import os


class TestSystem(unittest.TestCase):

    def setUp(self):
        # Example YAML data to simulate a system file
        self.example_yaml = """
        system_id: test_system
        description: Test system for validation
        detector:
          px_size: 0.1
          ff: 0.8
          thickness: 500
          trapping_depth: 5
        source:
          target_angle: 10
          target: Mo
          SID: 90
          filter: [(Be, 1.0), (Rh, 0.057)]
          external_filter: [(Al, 1.0)]
        """
        # Save it to a file to simulate loading from disk
        with open('test_system.yaml', 'w') as f:
            f.write(self.example_yaml)

    def test_initialization(self):
        # Initialize the System object
        system = System('test_system.yaml')

        # Test if the system attributes are correctly loaded
        self.assertEqual(system.system_id, 'test_system')
        self.assertEqual(system.description, 'Test system for validation')

        # Test if detector parameters are correctly parsed
        self.assertEqual(system.detector['px_size'], 0.1)
        self.assertEqual(system.detector['ff'], 0.8)
        self.assertEqual(system.detector['thickness'], 500)
        self.assertEqual(system.detector['trapping_depth'], 5)

        # Test if the source parameters are correctly parsed
        self.assertEqual(system.source['target_angle'], 10)
        self.assertEqual(system.source['target'], 'Mo')
        self.assertEqual(system.source['SID'], 90)

        # Test if filters are properly formatted as tuples
        self.assertEqual(system.source['filter'], [('Be', 1), ('Rh', 0.057)])
        self.assertEqual(system.source['external_filter'], [('Al', 1.0)])

        os.remove("test_system.yaml")


if __name__ == '__main__':
    unittest.main()
