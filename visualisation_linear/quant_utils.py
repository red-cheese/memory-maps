

import numpy as np


def rec_proba(inputs, output_mus, output_logsigmas_2):
    """Computes reconstruction probability as per http://dm.snu.ac.kr/static/docs/TR/SNUDM-TR-2015-03.pdf"""

    return np.sum(
            # logsigmas - matrix of shape (dataset size, input dim = 80),
            # as currently we're assuming a diagonal covariance matrix
            -(0.5 * np.log(2 * np.pi) + 0.5 * np.asarray(output_logsigmas_2))
            - 0.5 * (np.square(inputs - output_mus) / np.exp(output_logsigmas_2)),
            axis=1)
