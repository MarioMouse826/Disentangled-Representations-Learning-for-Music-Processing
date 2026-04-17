import numpy as np
from sklearn.ensemble import RandomForestClassifier


def _importance_matrix(z_means, factors_dict):
    """
    Train a random forest per factor, return importance matrix R.
    R[i, k] = importance of latent dim i for predicting factor k
    """
    latent_dim = z_means.shape[1]
    factor_names = list(factors_dict.keys())
    R = np.zeros((latent_dim, len(factor_names)))

    for k, name in enumerate(factor_names):
        factors = factors_dict[name]
        mask = factors != -1
        z = z_means[mask]
        f = factors[mask]

        if len(np.unique(f)) < 2:
            continue

        clf = RandomForestClassifier(n_estimators=100, random_state=42)
        clf.fit(z, f)
        R[:, k] = clf.feature_importances_

    return R, factor_names


def _entropy(probs):
    probs = probs[probs > 0]
    return -np.sum(probs * np.log(probs))


def dci(z_means, factors_dict):
    """
    Compute DCI disentanglement metrics.

    Args:
        z_means: np.ndarray (N, latent_dim)
        factors_dict: dict mapping factor name to np.ndarray (N,) of labels
                      e.g. {'pitch': pitch_array, 'identity': identity_array}
                      use -1 for missing labels

    Returns:
        dict with keys: disentanglement, completeness, informativeness
    """
    R, factor_names = _importance_matrix(z_means, factors_dict)
    K = len(factor_names)
    Z = z_means.shape[1]

    # row normalize for disentanglement
    row_sums = R.sum(axis=1, keepdims=True)
    P = np.divide(R, row_sums, where=row_sums > 0, out=np.zeros_like(R))

    # col normalize for completeness
    col_sums = R.sum(axis=0, keepdims=True)
    Q = np.divide(R, col_sums, where=col_sums > 0, out=np.zeros_like(R))

    # disentanglement
    d_scores = []
    weights = []
    for i in range(Z):
        if row_sums[i, 0] == 0:
            continue
        h = _entropy(P[i])
        d_i = 1.0 - h / np.log(K + 1e-10)
        d_scores.append(d_i)
        weights.append(row_sums[i, 0])
    D = np.average(d_scores, weights=weights) if d_scores else 0.0

    # completeness
    c_scores = []
    for k in range(K):
        h = _entropy(Q[:, k])
        c_k = 1.0 - h / np.log(Z + 1e-10)
        c_scores.append(c_k)
    C = np.mean(c_scores) if c_scores else 0.0

    # informativeness — accuracy per factor
    i_scores = []
    for name, factors in factors_dict.items():
        mask = factors != -1
        z = z_means[mask]
        f = factors[mask]
        if len(np.unique(f)) < 2:
            continue
        clf = RandomForestClassifier(n_estimators=100, random_state=42)
        clf.fit(z, f)
        i_scores.append(clf.score(z, f))
    I = np.mean(i_scores) if i_scores else 0.0

    return {"disentanglement": D, "completeness": C, "informativeness": I}