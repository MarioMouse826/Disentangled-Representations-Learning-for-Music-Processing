import numpy as np
from sklearn.feature_selection import mutual_info_classif


def entropy(labels):
    _, counts = np.unique(labels, return_counts=True)
    probs = counts / counts.sum()
    return -np.sum(probs * np.log(probs + 1e-10))


def mig(z_means, factors):
    """
    Compute Mutual Information Gap.

    Args:
        z_means: np.ndarray of shape (N, latent_dim) — encoder mean vectors
        factors: np.ndarray of shape (N,) — ground truth labels (e.g. pitch or identity)

    Returns:
        float — MIG score between 0 and 1
    """
    # filter out samples with missing labels (-1 sentinel)
    mask = factors != -1
    z_means = z_means[mask]
    factors = factors[mask]

    # MI between each latent dim and the factor
    # mutual_info_classif expects (N, n_features), factors as target
    mi = mutual_info_classif(z_means, factors, discrete_features=False)

    # sort descending
    mi_sorted = np.sort(mi)[::-1]

    if len(mi_sorted) < 2:
        return 0.0

    h = entropy(factors)
    if h == 0:
        return 0.0

    return (mi_sorted[0] - mi_sorted[1]) / h