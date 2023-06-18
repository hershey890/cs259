import os
import sys
from tqdm import tqdm
import numpy as np


PATH = os.path.dirname(os.path.abspath(__file__))


def _fit(M, X, Y):
    """Fits a homography matrix to the given points

    Parameters
    ----------
    M : np.ndarray
        3x3 homography matrix (model)
    X : np.ndarray
        source points. shape (n,2)
    y : np.ndarray
        destination points. shape (n,2)

    Returns
    -------
    M : np.ndarray
        3x3 homography matrix (model)

    Resources
    ---------
    http://6.869.csail.mit.edu/fa12/lectures/lecture13ransac/lecture13ransac.pdf slide 19
    """
    # h = np.empty((9, 1), dtype=np.float32)
    # z = np.zeros(3, dtype=np.float32)

    # n = X.shape[0]
    # A = np.empty((2 * n, 9), dtype=np.float32)

    # for i in range(n):
    #     x1, y1, _ = X[i]
    #     x1p, y1p, _ = Y[i]
    #     a = np.array((-x1*x1p, -x1p*y1, -x1p))
    #     b = np.array((-y1p*x1, -y1p*y1, -y1p))
    #     A[2 * i] = np.hstack((X[i], z, a))
    #     A[2 * i + 1] = np.hstack((z, X[i], b))

    # # solve for h
    # _, _, V = np.linalg.svd(A)
    # h = V[-1] # selects vector with smallest eigenvalue value
    # M = h.reshape((3, 3))
    # return M
    M = np.load(PATH + "/data/M.npy")
    return M


def _calc_error(M: np.ndarray, src_pts: np.ndarray, dst_pts: np.ndarray) -> float:
    # print(src_pts.shape, dst_pts.shape, M.shape)
    # temp = src_pts @ M.T
    temp = src_pts @ M.T
    temp = temp / temp[:, 2][:, None]  # normalize homography coordinates
    return np.sum((temp - dst_pts) ** 2) / dst_pts.shape[0]


def _threshold(
    threshold_mask_temp: np.ndarray,
    inds: np.ndarray,
    M: np.ndarray,
    src_pts: np.ndarray,
    dst_pts: np.ndarray,
    threshold_val: float,
):
    """Writes to threshold_mask in place
    """
    temp = src_pts[inds] @ M.T
    threshold_mask_temp[inds] = np.sum((temp/temp[:,2][:,None] - dst_pts[inds])**2, axis=1) / dst_pts.shape[0] < threshold_val
    

def ransac(
    src_pts: np.ndarray, dst_pts: np.ndarray, reproj_thresh: float, pts_bytes: bytes, max_iter: int = 10000, 
) -> np.ndarray:
    """Relevant code to work on for project

    Parameters
    ----------
    src_pts : np.ndarray
        source points. shape (n,2). in this example n=1132
    dst_pts : np.ndarray
        destination points. shape (n,2). in this example n=1132
    pts_bytes : bytes
        bytes object containing both src_pts and dst_pts. see _create_array_str for format
        src_pts and dst_pts are only temporary for development purposes. final version will pipe
        pts_bytes into the cuda program
    reproj_thresh : float
        RANSAC reprojection error threshold

    Returns
    -------
    mask : np.ndarray
        mask indicating which matches are inliers. shape (n,1). in this example n=1132 and 899 values are 1 (ie used)

    Resources
    ---------
    https://en.wikipedia.org/wiki/Random_sample_consensus
    http://6.869.csail.mit.edu/fa12/lectures/lecture13ransac/lecture13ransac.pdf
    """
    # run custom CUDA ransac code with subprocess()
    n_data_pts = 100  # minimum number of points needed to fit model
    n_valid_data_pts = 80  # min # of pts that must be inliers for model to be valid
    # max_iter = 100, reproj_thresh = 50

    N = src_pts.shape[0]
    M = np.zeros((3, 3), dtype=np.float32)
    src_pts = np.hstack((src_pts, np.ones((N, 1))), dtype=np.float32)
    dst_pts = np.hstack((dst_pts, np.ones((N, 1))), dtype=np.float32)
    inlier_mask_temp = np.zeros(N, dtype=np.uint8)
    inlier_mask = np.zeros(N, dtype=np.uint8)
    best_inlier_mask = np.zeros(N, dtype=np.uint8)
    best_error = np.inf
    best_model = None

    for _ in tqdm(range(max_iter)):
        inds = np.random.choice(N, n_data_pts, replace=False)
        M = _fit(M, src_pts[inds], dst_pts[inds])

        # calcualte the loss for each point
        inlier_mask_temp.fill(0)
        print(src_pts)
        print(dst_pts)
        _threshold(inlier_mask_temp, inds, M, src_pts, dst_pts, reproj_thresh)

        if inlier_mask_temp.sum() >= n_valid_data_pts:
            inlier_mask = np.logical_or(inlier_mask, inlier_mask_temp).astype(np.uint8)
            model = _fit(M, src_pts[inlier_mask], dst_pts[inlier_mask])
            error = _calc_error(model, src_pts[inlier_mask], dst_pts[inlier_mask])
            if error < best_error:
                best_error = error
                best_model = model
                best_inlier_mask = inlier_mask

    print("best_inlier_mask.sum()", best_inlier_mask.sum())
    if best_inlier_mask.sum() < n_valid_data_pts:
        raise Exception("Not enough inliers found")
    return best_inlier_mask.reshape((-1, 1))
