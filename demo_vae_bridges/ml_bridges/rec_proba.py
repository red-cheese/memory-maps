

import numpy as np


def rec_proba(inputs, output_mus, output_logsigmas_2):
    """
    Computes reconstruction probability (An and Cho, 2015).
    For convenience, only one sample was taken for this in the latent space (L = 1).
    """

    return np.sum(
            # output_logsigmas_2 - matrix of shape (dataset size, input dim),
            # as currently we're assuming a diagonal covariance matrix.
            -(0.5 * np.log(2 * np.pi) + 0.5 * np.asarray(output_logsigmas_2))
            - 0.5 * (np.square(inputs - output_mus) / np.exp(output_logsigmas_2)),
            axis=1)
