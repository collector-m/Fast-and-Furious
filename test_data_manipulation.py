import data_manipulation
import unittest
import numpy as np


class TestDataManipulation(unittest.TestCase):

    

    def test_create_occupancy_grids(self):
        point_cloud = np.array([[1, 1, 2], [1, 1, 4], [3, 4, 3], [3, 3, 4], [2, 3, 1]])
        img_h = 5
        img_w = 4
        z_min = 0
        z_max = 5
        num_slices = 4
        resolution = 1
        cloud_dict = {1: point_cloud}
        cloud_dict = data_manipulation.create_occupancy_grids(
            cloud_dict,
            resolution=resolution,
            Xsize=img_w,
            Ysize=img_h,
            bins=num_slices,
            minHeight=z_min,
            maxHeight=z_max,
        )
        print(cloud_dict[1].shape)
        self.assertEqual(None, None)


if __name__ == "__main__":
    unittest.main()
