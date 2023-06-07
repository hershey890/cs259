import os
import sys
import numpy as np


PATH = os.path.dirname(os.path.abspath(__file__))
    

# TODO: actually implement this, dont just return M_truth
M_truth = np.load(PATH + '/data/M.npy')
def _fit(M, X, y):
    """
    Parameters
    ----------
    M : np.ndarray
        3x3 homography matrix (model)
    X : np.ndarray
        source points. shape (n,2)
    y : np.ndarray
        destination points. shape (n,2)
    """
    return M_truth


def _calc_error(M: np.ndarray, src_pts: np.ndarray, dst_pts: np.ndarray) -> float:
    temp = src_pts @ M.T
    temp = temp / temp[2]
    return np.sum((temp - dst_pts)**2) / dst_pts.shape[0]
    # return np.sum((M @ X.T - y.T)**2) / y.shape[0]


def _threshold(threshold_mask_temp: np.ndarray, inds: np.ndarray, M: np.ndarray, src_pts: np.ndarray, dst_pts: np.ndarray, threshold_val: float) -> int:
    """Writes to threshold_mask in place

    Returns
    -------
    n_valid : int
        number of valid points (ie inliers)
    """
    """
    thresholded = (
            self.loss(y[ids][self.n :], maybe_model.predict(X[ids][self.n :]))
            < self.t
        )
    """
    n_valid = 0
    for i in inds:
        err = _calc_error(M, src_pts[i], dst_pts[i])
        # err = _calc_error(M, dst_pts[i], src_pts[i])
        if err < 100: # TODO: temporary for development
        # if err < threshold_val:
            n_valid += 1
            threshold_mask_temp[i] = 1
        else:
            threshold_mask_temp[i] = 0
    
    return n_valid


def ransac(src_pts: np.ndarray, dst_pts: np.ndarray, reproj_thresh: float, pts_bytes: bytes) -> np.ndarray:
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
    n_data_pts = 100 # minimum number of points needed to fit model
    n_valid_data_pts = 20 # min # of pts that must be inliers for model to be valid
    max_iter = 1000
    reproj_thresh = 10.0 # TODO: temp, delete later

    N = src_pts.shape[0]
    M = np.zeros((3, 3), dtype=np.float32)
    src_pts = np.hstack((src_pts, np.ones((N, 1))), dtype=np.float32)
    dst_pts = np.hstack((dst_pts, np.ones((N, 1))), dtype=np.float32)
    inlier_mask_temp = np.zeros(N, dtype=np.uint8)
    inlier_mask      = np.zeros(N, dtype=np.uint8)
    best_inlier_mask = np.zeros((N, 1), dtype=np.uint8)
    best_error = np.inf
    best_model = None

    for _ in range(max_iter):
        inds = np.random.choice(N, n_data_pts, replace=False)
        M = _fit(M, src_pts[inds], dst_pts[inds])

        # calcualte the loss for each point
        inlier_mask_temp.fill(0)
        n_valid = _threshold(inlier_mask_temp, inds, M, src_pts, dst_pts, reproj_thresh)

        if n_valid >= n_valid_data_pts:
            # print('WORKSSSSSSSSSSSSSSSSSSSSSSSSSSSS')
            inlier_mask = np.logical_or(inlier_mask, inlier_mask_temp)
            model = _fit(M, src_pts[inlier_mask], dst_pts[inlier_mask])
            error = _calc_error(model, src_pts[inlier_mask], dst_pts[inlier_mask])
            if error < best_error:
                best_error = error
                best_model = model
                best_inlier_mask = inlier_mask

    print("best_inlier_mask.sum()", best_inlier_mask.sum())
    if best_inlier_mask.sum() < n_valid_data_pts:
        raise Exception("Not enough inliers found")
    return best_inlier_mask.reshape((-1, 1)).astype(np.uint8)
    # return np.ones((N, 1), dtype=np.uint8)