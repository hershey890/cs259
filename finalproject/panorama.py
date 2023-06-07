"""
Resources
---------
https://medium.com/analytics-vidhya/panorama-formation-using-image-stitching-using-opencv-1068a0e8e47b
https://pyimagesearch.com/2016/01/11/opencv-panorama-stitching/
data: https://github.com/ppwwyyxx/OpenPano/releases/tag/0.1
"""
import sys
import os
import subprocess
from pathlib import Path
from typing import Tuple, List, Union, Dict
import numpy as np
import cv2
import matplotlib.pyplot as plt
PATH = os.path.dirname(os.path.abspath(__file__))
sys.path.append(PATH)
import importlib
from ransac import ransac
# importlib.reload(sys.modules['ransac'])


WRITE_VALUES_TO_FILE = False
ENDIAN = 'little' # 'little' or 'big'


def _create_array_str(src_pts: np.ndarray, dst_pts: np.ndarray) -> bytes:
    """Writes src_pts or dst_pts to a bytes object

    CUDA code will use piping in the final version (or a direct Cython call) but
    this is useful for development.

    Parameters
    ----------
    src_pts : np.ndarray
        array to write to file. Shape (n,2). n~1000
    dst_pts: np.ndarray
        also wrriten to in file

    Bytes Format
    -----------
    The file stores both src_pts and dst_pts in the same file. 
    The number of points is n*2 where n = src_pts.shape[0]
    The data type is float32
    No newlines

    4 bytes: number of points for a single array (there are 2 arrays) (2*n) 
    n*2*sizeof(float32) bytes for src_pts. Each pair of bytes corresponds to one point (x,y) 
    n*2*sizeof(float32) bytes for dst_pts. Each pair of bytes corresponds to one point (x,y)

    Word lengths (no newlines)
    4 bytes
    n*2n*4 bytes
    n*2n*4 bytes

    Example:
    [byte 1][byte 2][byte 3][byte 4]
    [byte 1 for src_pts x][byte 2 for src_pts x][byte 3 for src_pts x][byte 4 for src_pts x] ... [byte 2*n-1 for src_pts x][byte 2*n for src_pts x]
    [byte 1 for src_pts y][byte 2 for src_pts y][byte 3 for src_pts y][byte 4 for src_pts y] ... [byte 2*n-1 for src_pts y][byte 2*n for src_pts y]
    """
    assert src_pts.shape == dst_pts.shape
    assert src_pts.dtype == dst_pts.dtype and src_pts.dtype == np.float32

    src_pts = src_pts.flatten()
    dst_pts = dst_pts.flatten()

    # convert values to bytes
    sizeof_float32 = 4
    n_bytes = src_pts.shape[0] * sizeof_float32
    if ENDIAN == 'little':
        n_bytes_bytes = n_bytes.to_bytes(4, byteorder='little')
        src_pts_bytes = src_pts.astype('<f').tobytes()
        dst_pts_bytes = dst_pts.astype('<f').tobytes()
    elif ENDIAN == 'big':
        n_bytes_bytes = n_bytes.to_bytes(4, byteorder='big')
        src_pts_bytes = src_pts.astype('>f').tobytes()
        dst_pts_bytes = dst_pts.astype('>f').tobytes()
    else:
        raise ValueError('ENDIAN must be one of {little, big}')

    return n_bytes_bytes + src_pts_bytes + dst_pts_bytes


def _find_matches(img_left, img_right) -> Tuple[Tuple[cv2.KeyPoint], Tuple[cv2.KeyPoint], List[cv2.DMatch]]:
    """OpenCV Code
    """
    # Open images and detect keypoints
    gray = cv2.cvtColor(img_right, cv2.COLOR_BGR2GRAY)
    sift = cv2.xfeatures2d.SIFT_create()
    kp_right, des_right = sift.detectAndCompute(gray, None) # len 3655, des1.shape 3655, 128
    gray = cv2.cvtColor(img_left, cv2.COLOR_BGR2GRAY)
    kp_left, des_left = sift.detectAndCompute(gray, None) # len 5618
    
    # match keypoints
    matcher = cv2.BFMatcher()
    matches = matcher.knnMatch(des_right, des_left, k=2)

    # filter matches
    good_matches = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good_matches.append(m)

    return kp_left, kp_right, good_matches


def _find_homography(
        kp1: List[cv2.KeyPoint], 
        kp2: List[cv2.KeyPoint], 
        good_matches: List[cv2.DMatch],
        ransac_reproj_thresh: float = 10.0,
        ransac_max_iter: int = 10000,
        ransac_method: str='opencv', 
        write_values_to_file: bool = False
    ) -> Tuple[np.ndarray, np.ndarray]:
    """
    Parameters
    ----------
    kp1 : List[cv2.KeyPoint]
        keypoints from image 1
    kp2 : List[cv2.KeyPoint]
        keypoints from image 2
    good_matches : List[cv2.DMatch]
        list of good matches between keypoints from image 1 and image 2
    ransac_method : {'opencv', 'cuda'}
        method to use for RANSAC
    write_values_to_file : bool
        if True, writes src_pts, dst_pts, and mask to file. Useful for development purposes.

    Returns
    -------
    M : np.ndarray
        3x3 homography matrix
    mask : np.ndarray
        mask indicating which matches are inliers
    """
    min_match_count = 10

    if len(good_matches) <= min_match_count:
        raise Exception('Not enough matches are found - {}/{}'.format(len(good_matches), min_match_count))
    kp1_np = np.float32([x.pt for x in kp1])
    kp2_np = np.float32([x.pt for x in kp2])
    src_pts = np.float32([kp1_np[m.queryIdx] for m in good_matches])
    dst_pts = np.float32([kp2_np[m.trainIdx] for m in good_matches])
    pts_bytes = _create_array_str(src_pts, dst_pts)

    # for development purposes
    if write_values_to_file:
        with open('./data/src_dst_pts.bin', 'wb') as f:
            f.write(pts_bytes)

    if ransac_method == 'opencv':
        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, ransac_reproj_thresh)
        if write_values_to_file:
            with open(PATH + '/data/mask.bin', 'wb') as f:
                f.write(mask[:,0].tobytes())
            with open(PATH + '/data/M.bin', 'wb') as f:
                f.write(M.tobytes())
                
    elif ransac_method == 'cuda':
        mask = ransac(src_pts, dst_pts, ransac_reproj_thresh, pts_bytes, max_iter=10000)
        src_pts = src_pts[mask[:, 0] == 1]
        dst_pts = dst_pts[mask[:, 0] == 1]
        M, _ = cv2.findHomography(src_pts, dst_pts, 0, ransac_reproj_thresh) # without RANSAC
    else:
        raise ValueError('ransac_method must be one of {opencv, cuda}')
    assert mask.dtype == np.uint8

    if write_values_to_file:
        np.save(PATH + '/data/M.npy', M)
        np.save(PATH + '/data/mask.npy', mask)
        np.save(PATH + '/data/src_pts.npy', src_pts)
        np.save(PATH + '/data/dst_pts.npy', dst_pts)
    
    return M.astype(np.float32), mask[:, 0]


def _plot(data_folder: str, subplot_idx: int, img: np.ndarray, mode: str, filename: str = None):
    """Plots or saves image depending on `mode`
    """
    if mode == 'plot':
        plt.subplot(subplot_idx)
        plt.imshow(img[:,:,::-1])
    elif mode == 'save':
        cv2.imwrite(data_folder + '/data/' + filename, img)


def main(data_folder: str | Path, ransac_method: str, ransac_reproj_thresh: float = 10.0, ransac_max_iter: int = 10000, mode: str = 'plot'):
    """
    Parameters
    ----------
    data_folder : str | Path
        Path to folder containing medium02 and medium03 images
    ransac_reproj_thresh : float
        RANSAC reprojection error threshold
    plot_mode : {'plot', 'save', None}
        indicates if images should be displayed or saved in `data_folder`
        If saved name follows the format `output_{i}_*.png`
        if None does not save or plot images
    """

    if mode is not None:
        assert mode in ['plot', 'save']
        plt.figure(figsize=(10, 20))

    if isinstance(data_folder, Path):
        data_folder = str(data_folder)
    img_right = cv2.imread(data_folder + '/data/medium03.jpg')
    img_left = cv2.imread(data_folder + '/data/medium02.jpg')
    # plot both images
    # plt.figure(figsize=(10, 20))
    # plt.subplot(121)
    # plt.imshow(img_left[:,:,::-1])
    # plt.subplot(122)
    # plt.imshow(img_right[:,:,::-1])
    # plt.show()

    # Detect keypoints and find matches between keypoints
    kp_left, kp_right, good_matches = _find_matches(img_left, img_right)

    # find homography
    M, mask = _find_homography(kp_right, kp_left, good_matches, ransac_reproj_thresh, ransac_max_iter, ransac_method=ransac_method,  write_values_to_file=WRITE_VALUES_TO_FILE)
    h, w, d = img_right.shape
    pts = np.float32([[0, 0], [0, h-1], [w-1, h-1], [w-1, 0]]).reshape(-1, 1, 2)
    dst = cv2.perspectiveTransform(pts, M).astype(np.int32)
    img_right = cv2.polylines(img_right, [dst], True, 255, 3, cv2.LINE_AA)
    _plot(data_folder, 411, img_right, mode, 'output_1_stitched.jpg')

    # draw matches
    draw_params = dict(matchColor=(0, 255, 0), singlePointColor=None, matchesMask=mask, flags=2)
    img3 = cv2.drawMatches(img_left, kp_right, img_right, kp_left, good_matches, None, **draw_params)
    _plot(data_folder, 412, img3, mode, 'output_2_matches.jpg')

    # stitch images
    dst = cv2.warpPerspective(img_right, M, (img_right.shape[1] + img_left.shape[1], img_right.shape[0]))
    dst[0:img_right.shape[0], 0:img_right.shape[1]] = img_left
    _plot(data_folder, 413, dst, mode, 'output_3_stitched.jpg')


if __name__ == '__main__':
    main(PATH + '/data/', ransac_method='cuda', ransac_reproj_thresh=10.0, ransac_max_iter=10000, mode=None)