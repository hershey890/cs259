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


def _threshold(threshold_mask: np.ndarray, M: np.ndarray, X: np.ndarray, y: np.ndarray, threshold_val: float) -> int:
    """Writes to threshold_mask in place

    Returns
    -------
    n_valid : int
        number of valid points (ie inliers)
    """
    n_valid = threshold_mask.shape[0]
    threshold_mask.fill(1)
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
    # TODO: implement this in Python first. 
    # follow RANSAC example http://6.869.csail.mit.edu/fa12/lectures/lecture13ransac/lecture13ransac.pdf
    # will have to implement calculating M matrix
    
    
    
    """
    for _ in range(self.k):
        ids = rng.permutation(X.shape[0])

        maybe_inliers = ids[: self.n]
        maybe_model = copy(self.model).fit(X[maybe_inliers], y[maybe_inliers])

        thresholded = (
            self.loss(y[ids][self.n :], maybe_model.predict(X[ids][self.n :]))
            < self.t
        )

        inlier_ids = ids[self.n :][np.flatnonzero(thresholded).flatten()]

        if inlier_ids.size > self.d:
            inlier_points = np.hstack([maybe_inliers, inlier_ids])
            better_model = copy(self.model).fit(X[inlier_points], y[inlier_points])

            this_error = self.metric(
                y[inlier_points], better_model.predict(X[inlier_points])
            )

            if this_error < self.best_error:
                self.best_error = this_error
                self.best_fit = maybe_model

    return self
    """
    n_data_pts = 10 # minimum number of points needed to fit model
    n_valid_data_pts = 10 # min # of pts that must be inliers for model to be valid
    max_iter = 100

    N = src_pts.shape[0]
    M = np.zeros((3, 3), dtype=np.float32)
    src_pts = np.hstack((src_pts, np.ones((N, 1))), dtype=np.float32)
    dst_pts = np.hstack((dst_pts, np.ones((N, 1))), dtype=np.float32)
    threshold_mask = np.zeros(N, dtype=np.uint8)
    inlier_pts = np.zeros((N, 1), dtype=np.uint8)
    for _ in range(max_iter):
        inds = np.random.choice(N, n_data_pts, replace=False)
        M = _fit(M, src_pts[inds], dst_pts[inds])

        # calcualte the loss for each point
        n_valid = _threshold(threshold_mask, M, src_pts, dst_pts, reproj_thresh)

        if n_valid >= n_valid_data_pts:
            pass

    
    return np.ones((N, 1), dtype=np.uint8)