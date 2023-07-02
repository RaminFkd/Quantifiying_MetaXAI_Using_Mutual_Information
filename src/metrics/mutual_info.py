
from typing import Callable, Dict, List, Tuple
import numpy as np

from scipy.special import digamma
from scipy.stats import differential_entropy, gaussian_kde

def _get_k_nearest_neighbours(
    x: List[float] | np.ndarray,
    y: List[float] | np.ndarray,
    k: int,
    eps: float = 1e-15,
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

    #k_nn_in_xx = np.argsort(xx_dist)[:, 1:k+1]
    k_nn_in_xx = np.argpartition(xx_dist, k, axis=1)[:, k:k+1]

    dist_to_knn = xx_dist[np.arange(len(x)), k_nn_in_xx.squeeze()]
    dist_to_knn = np.where(dist_to_knn == 0.0, eps, dist_to_knn)

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
    k: int,
    eps: float = 1e-15,
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
    eps : float
        The epsilon value to use for numerical stability
    """

    x, y = norm_distribution(x, y, eps)

    n_xy = _get_k_nearest_neighbours(x, y, k, eps)
    n_yx = _get_k_nearest_neighbours(y, x, k, eps)

    n = n_xy['N_x'] + n_xy['N_y']

    i_xy = digamma(n) + digamma(k) - digamma(n_xy['m_i']) - digamma(n_xy['N_x'])
    i_yx = digamma(n) + digamma(k) - digamma(n_yx['m_i']) - digamma(n_yx['N_y'])

    I = np.concatenate([i_xy, i_yx])

    if I.max() - I.min() == 0:
        I = (I - I.min()) / (I.max() - I.min() + eps)
    else:
        I = (I - I.min()) / (I.max() - I.min())

    return {
        'I': I,
        'MI': I.mean(),
    }


def continuous_mutual_info_gap(
    x: List[float] | np.ndarray,
    y: List[float] | np.ndarray,
    k: int,
    eps: float = 1e-15
):
    """
    Computes the continuous mutual information gap between two variables
    using the continuous mutual information score and the differential entropy
    of the two variables.

    Parameters
    ----------
    x : List[float] | np.ndarray
        The first variable
    y : List[float] | np.ndarray
        The second variable
    k : int
        The k nearest neighbours for the continuous mutual information score
    eps : float, optional
        A small value to guarantee stability, by default 1e-15

    Returns
    -------
    Dict[str, float]
        A dictionary containing the continuous mutual information gap for the
        two variables with the keys 'MI_x', 'MI_y' and the continuous mutual
        information 'MI'
    """
    mi_score = continuous_mutual_info_score(x, y, k, eps)
    sorted_mi = np.sort(mi_score['I'])
    max_mi_0 = sorted_mi[-1]
    max_mi_1 = sorted_mi[-2]
    for i in range(3, len(sorted_mi)):
        if sorted_mi[-i] < max_mi_0:
            max_mi_1 = sorted_mi[-i]
            break


    x, y = norm_distribution(x, y, eps)
    entropy_x = differential_entropy(x.squeeze())
    entropy_y = differential_entropy(y.squeeze())

    return {
        'MI': mi_score['MI'],
        'MIG_x': np.abs((max_mi_0 - max_mi_1) / entropy_x),
        'MIG_y': np.abs((max_mi_0 - max_mi_1) / entropy_y),
    }


def norm_distribution(
    x: List[float] | np.ndarray,
    y: List[float] | np.ndarray,
    eps: float = 1e-15
) -> Tuple:
    """
    Returns the normalized distributions of the two variables

    Parameters
    ----------
    x : List[float] | np.ndarray
        The first variable
    y : List[float] | np.ndarray
        The second variable
    eps : float, optional
        A small value to guarantee stability, by default 1e-15

    Returns
    -------
    Tuple
        The normalized distributions of the two variables
    """
    if isinstance(x, list):
        x = np.array(x)[:, np.newaxis]
    if isinstance(y, list):
        y = np.array(y)[:, np.newaxis]

    x, y = x.astype(np.float64), y.astype(np.float64)

    if len(x.shape) == 1:
        x = x[:, np.newaxis]
    if len(y.shape) == 1:
        y = y[:, np.newaxis]

    x = (x - x.min()) / (x.max() - x.min())
    y = (y - y.min()) / (y.max() - y.min())

    x = np.where(x == 0.0, eps, x)
    y = np.where(y == 0.0, eps, y)

    x = np.where(x == 1.0, 1.0 - eps, x)
    y = np.where(y == 1.0, 1.0 - eps, y)

    return x, y
