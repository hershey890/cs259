M.bin File Schema
-----------------
3x3 matrix of float32's
each row is stored continuously in memory

src_dst_pts.bin File Schema
---------------------------
see panorama.py _create_array_str

mask.bin File Schema
--------------------
1d array of uint8_t values

TODO:
-----
- add a method of setting cuda or opencv method from the main() function
- code vanilla RANSAC in python