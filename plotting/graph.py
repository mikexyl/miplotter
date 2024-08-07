import pandas as pd
import logging
import numpy as np

logger = logging.getLogger(__name__)


class Graph:
    def __init__(self) -> None:
        pass


def align_trajectories(traj1, traj2, keys1, keys2):
    keys1 = np.array(keys1)
    keys2 = np.array(keys2)

    # Find the common keys
    common_keys = np.intersect1d(keys1, keys2)

    # Extract corresponding points based on the common keys
    indices1 = [np.where(keys1 == key)[0][0] for key in common_keys]
    indices2 = [np.where(keys2 == key)[0][0] for key in common_keys]

    points1 = traj1[indices1, :]
    points2 = traj2[indices2, :]

    # Translate points to have their centroids at the origin
    centroid1 = np.mean(points1, axis=0)
    centroid2 = np.mean(points2, axis=0)
    centered_points1 = points1 - centroid1
    centered_points2 = points2 - centroid2

    # Compute the optimal rotation matrix using Singular Value Decomposition (SVD)
    U, S, Vt = np.linalg.svd(np.dot(centered_points1.T, centered_points2))
    R = np.dot(U, Vt)

    # Apply the rotation to centered_points2
    aligned_points2 = np.dot(centered_points2, np.linalg.inv(R))

    # Calculate the sum of squared distances (SSD) between aligned points
    ssd = np.sum((centered_points1 - aligned_points2) ** 2)

    # Translate aligned_points2 back to match the original position of points1
    aligned_points2 += centroid1

    # rmse
    rmse = np.sqrt(ssd / len(common_keys))

    return aligned_points2, R, centroid1, rmse


def apply_alignment(traj2, R, centroid1, centroid2):
    # Ensure traj2 is a 2D array
    if isinstance(traj2, (pd.DataFrame, pd.Series)):
        traj2 = traj2.to_numpy()

    traj2 = np.atleast_2d(traj2)

    # Translate traj2 to the origin based on its centroid
    traj2_centered = traj2 - centroid2

    # Apply the rotation matrix R
    traj2_rotated = np.dot(traj2_centered, R)

    # Translate the rotated traj2 to the centroid of traj1
    traj2_aligned = traj2_rotated + centroid1

    return traj2_aligned


def loadTxt(file, has_keys=False):
    # open the file and check if it's empty
    with open(file, "r") as f:
        if not f.read(1):
            # warn
            return [], np.array([])

    # open txt file and load it with pandas
    df = pd.read_csv(file, delimiter=" ", header=None)
    # add a column with the keys
    if not has_keys:
        df.insert(0, "keys", range(0, df.shape[0]))
    print(df.shape)
    return np.array(df.iloc[:, 0]), np.array(df.iloc[:, 1:3])


def loadCSV(file):
    # open the file and check if it's empty
    with open(file, "r") as f:
        if not f.read(1):
            # warn
            return [], np.array([])

    # open csv file and load it with pandas
    df = pd.read_csv(file)
    # remove lines if the first column doesn't start with 'p'
    df = df[df.iloc[:, 0].str.startswith("p")]
    # remove the 'p' letter of the first column
    df.iloc[:, 0] = df.iloc[:, 0].str.replace("p", "")
    # convert the first column to int
    df.iloc[:, 0] = df.iloc[:, 0].astype(int)
    return np.array(df.iloc[:, 0]), np.array(df.iloc[:, 1:])


def compareGraphs(csv1, csv2):
    # load the csv files
    loadCSV(csv1)
    loadCSV(csv2)
