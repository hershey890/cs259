import sys
import os
import unittest
import numpy as np
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../')
from panorama import _create_array_str


class TestCase(unittest.TestCase):

    def test_create_array_str(self):
        # yeah ik this is slightly badly written
        src_pts_truth = np.random.random((10, 2)).astype(np.float32)
        dst_pts_truth = np.random.random((10, 2)).astype(np.float32)

        byte_arr = _create_array_str(src_pts_truth, dst_pts_truth)
        n_bytes = int.from_bytes(byte_arr[:4], byteorder='little')
        src_pts_bytes = byte_arr[4:4+n_bytes]
        dst_pts_bytes = byte_arr[4+n_bytes:]

        src_pts_test = np.frombuffer(src_pts_bytes, dtype=np.float32).reshape(-1, 2)
        dst_pts_test = np.frombuffer(dst_pts_bytes, dtype=np.float32).reshape(-1, 2)
        self.assertTrue(np.equal(src_pts_test, src_pts_truth).all())
        self.assertTrue(np.equal(dst_pts_test, dst_pts_truth).all())

        


if __name__ == '__main__':
    unittest.main()