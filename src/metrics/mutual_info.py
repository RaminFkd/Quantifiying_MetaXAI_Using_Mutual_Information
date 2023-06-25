
from typing import Dict, List
import numpy as np

from scipy.special import digamma


def _get_k_nearest_neighbours(
    x: List[float] | np.ndarray,
    y: List[float] | np.ndarray,
    k: int
) -> Dict[str, np.ndarray]:

    # Compute the distance matrix between x and y using the euclidean distance
    # We use a vectorized approach based on the the binomial out multiplication
    # (a-b)^2 = a^2 + b^2 - 2ab
    xy_dist = np.einsum('ij,ij->i', x, x)[:, np.newaxis] \
        + np.einsum('ij,ij->i', y, y)[np.newaxis, :] - 2 * x @ y.T
    xy_dist = np.sqrt(xy_dist)

    xx_dist = np.einsum('ij,ij->i', x, x)[:, np.newaxis] \
        + np.einsum('ij,ij->i', x, x)[np.newaxis, :] - 2 * x @ x.T
    xx_dist = np.sqrt(xx_dist)

    k_nn_in_xx = np.argsort(xx_dist)[:, k]
    dist_to_knn = xx_dist[np.arange(len(x)), k_nn_in_xx.squeeze()]

    xxy_dist = np.concatenate([xy_dist, xx_dist], axis=1)
    m_i = np.where(xxy_dist < dist_to_knn[:, np.newaxis], 1, 0).sum(axis=1)

    return {
        'N_x': len(x),
        'N_y': len(y),
        'm_i': m_i,
        'd_i': dist_to_knn,
    }


def continuous_mutual_info_score(
    x: List[float] | np.ndarray,
    y: List[float] | np.ndarray,
    k: int
) -> float:
    """
    Based on the paper "Mutual information between discrete and continuous
    data-sets" by Ross, Brian C. (2014)

    Parameters
    ----------
    x :
        The first variable
    y :
        The second variable
    k : int
        The k nearest neighbours
    """
    if isinstance(x, list):
        x = np.array(x)[:, np.newaxis]
    if isinstance(y, list):
        y = np.array(y)[:, np.newaxis]

    if len(x.shape) == 1:
        x = x[:, np.newaxis]
    if len(y.shape) == 1:
        y = y[:, np.newaxis]

    n_xy = _get_k_nearest_neighbours(x, y, k)
    n_yx = _get_k_nearest_neighbours(y, x, k)

    n = n_xy['N_x'] + n_xy['N_y']
    i_xy = digamma(n) + digamma(k) - digamma(n_xy['m_i']) - digamma(n_xy['N_x'])
    i_yx = digamma(n) + digamma(k) - digamma(n_yx['m_i']) - digamma(n_yx['N_y'])
    I = np.concatenate([i_xy, i_yx])
    I = (I - I.min()) / (I.max() - I.min())
    return I.mean()
