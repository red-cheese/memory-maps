import matplotlib.pyplot as plt
import numpy as np
import os
import pickle


N = 150000
H = 0.01  # Linspace step
DATA_DIR = './data/synthetic_samples'


class Params:
    def __init__(self, id_, dim, k, sigma,
                 m=None, l=None, mu_shift=None, sigma_shift=None, results_dir=None):
        """
        :param id_: Dataset ID
        :param dim: Data dimensionality
        :param k: Trend angle
        :param sigma: Gaussian noise sigma
        :param m: Number of anomalies
        :param l: Length of each anomaly (number of ticks)
        :param mu_shift: Gaussian anomaly shift mean
        :param sigma_shift: Gaussian anomaly shift std
        """

        self.id_ = id_
        # TODO Different ks for different dims?
        self.dim = dim
        self.k = k
        self.sigma = sigma

        self.m = m
        self.l = l
        self.mu_shift = mu_shift
        self.sigma_shift = sigma_shift

        self.anom = self.m is not None
        self.name = 'anom{anom}_id{id}_size{N}_dim{dim}_k{k}_sigma{sigma}'.format(
                anom=self.anom, id=self.id_, N=N, dim=self.dim, k=self.k, sigma=self.sigma)
        self.title = 'Anomaly: {anom} ; ID: {id} ; size: {N}\n' \
            'dim: {dim} ; k: {k} ; sigma: {sigma}'.format(
                anom=self.anom, id=self.id_, N=N, dim=self.dim, k=self.k, sigma=self.sigma)
        if self.anom:
            self.name += '_m{m}_l{l}_mushift{mu_shift}_sigmashift{sigma_shift}'.format(
                m=self.m, l=self.l, mu_shift=self.mu_shift, sigma_shift=self.sigma_shift)
            self.title += '\nm: {m} ; l: {l} ; mu_shift: {mu_shift} ; sigma_shift: {sigma_shift}'.format(
                m=self.m, l=self.l, mu_shift=self.mu_shift, sigma_shift=self.sigma_shift)

        self.dir = results_dir


# class SineParams:
#     def __init__(self, id_, dim, A, w, phi, sigma, results_dir=None):
#         self.id_ = id_
#         self.dim = dim
#         self.A = A
#         self.w = w
#         self.phi = phi
#         self.sigma = sigma
#         self.anom = False
#         self.dir = results_dir
#
#         self.name = 'sine_anom{anom}_id{id}_size{N}_dim{dim}_A{A}_w{w}_phi{phi}_sigma{sigma}'.format(
#                 anom=self.anom, id=self.id_, N=N, dim=self.dim, A=self.A, w=self.w, phi=self.phi, sigma=self.sigma)
#         self.title = 'Sine wave - Anomaly: {anom} ; ID: {id} ; size: {N}\n' \
#                      'dim: {dim} ; A: {A} ; w: {w} ; phi: {phi} ; sigma: {sigma}'.format(
#                 anom=self.anom, id=self.id_, N=N, dim=self.dim, A=self.A, w=self.w, phi=self.phi, sigma=self.sigma)


def _generate_linear(params, plot=True, save=False):
    """
    Generate one run with a linear trend with no anomalies.
    """

    print('Generate:', params.name)

    xs = np.linspace(start=0, stop=N * H, num=N)
    assert len(xs) == N
    print('X[0]:', xs[0])
    print('X[-1]:', xs[-1])
    print('X[:20]:', xs[:20])

    noise = np.random.normal(0, params.sigma, size=(N, params.dim))
    ys = params.k * np.repeat(xs.reshape(-1, 1), params.dim, axis=1) + noise
    print('Y[0][0]:', ys[0, 0])
    print('Y[-1][0]:', ys[-1, 0])
    print('Y[:20][0]:', ys[:20, 0])

    # Plot component 0.
    if plot:
        plt.title(params.title)
        plt.plot(ys[:, 0])
        plt.savefig(os.path.join(DATA_DIR, params.name + '_line.png'))
        # plt.show()

    # TODO Save to .pkl file if requested

    print('Done Generate:', params.name)
    return ys


# def _generate_linear_anomalies(params, plot=True, save=False):
#     """
#     Generate one run with a linear trend with anomalies.
#     """
#
#     # TODO Fix for multiple dimensions
#     print('Generate:', params.name)
#
#     ys = _generate_linear(params, plot=False, save=False)
#
#     # Choose start indices for anomalies.
#     # Don't put an anomaly too close to the start or to the end.
#     all_idx = np.arange(1 * params.l, N - 2 * params.l)
#     np.random.shuffle(all_idx)
#     anomaly_idx = sorted(all_idx[:params.m])
#     print('Anomaly start indices:', anomaly_idx)
#
#     # Sample anomaly shifts.
#     # TODO Add smoothing
#     anomaly_shifts = np.random.normal(params.mu_shift, params.sigma_shift, size=params.m)
#     print('Anomaly shifts:', anomaly_shifts)
#
#     print()
#
#     # Now add anomaly shifts to the data and create labels.
#     labels = np.zeros(N, dtype=np.bool)
#     for i, shift in zip(anomaly_idx, anomaly_shifts):
#         ys[i:(i + params.l)] += shift
#         labels[i:(i + params.l)] = True
#
#     if plot:
#         plt.title(params.title)
#         idx_yes = np.where(labels)[0]
#         plt.scatter(idx_yes, ys[idx_yes], color='red', s=1)
#         idx_no = np.where(~labels)[0]
#         plt.scatter(idx_no, ys[idx_no], color='green', s=1)
#         plt.savefig(os.path.join(DATA_DIR, params.name + '_scatter.png'))
#         plt.show()
#
#         plt.title(params.title)
#         plt.plot(ys)
#         plt.savefig(os.path.join(DATA_DIR, params.name + '_line.png'))
#         plt.show()
#
#     # TODO Save if requested
#
#     print('Done Generate:', params.name)
#     return ys, labels


def split_train_test(data, labels, split=0.7):  # No shuffling.
    train_len = int(len(data) * split)
    train_data = data[:train_len]
    test_data = data[train_len:]
    train_labels = None if labels is None else labels[:train_len]
    test_labels = None if labels is None else labels[train_len:]
    return train_data, train_labels, test_data, test_labels


def generate(params, split=0.7, plot=True, save=False):
    # if params.anom:
    #     data, labels = _generate_linear_anomalies(params, plot=plot, save=save)
    # else:
    #     data, labels = _generate_linear(params, plot=plot, save=save), None
    data, labels = _generate_linear(params, plot=plot, save=save), None
    return split_train_test(data, labels, split=split)


# def generate_sine(params, split=0.7, plot=True, save=False):
#     print('Generate:', params.name)
#
#     xs = np.linspace(start=0, stop=N * H, num=N)
#     assert len(xs) == N
#     print('X[0]:', xs[0])
#     print('X[-1]:', xs[-1])
#     print('X[:20]:', xs[:20])
#
#     noise = np.random.normal(0, params.sigma, size=(N, params.dim))
#     repeat_xs = np.repeat(xs.reshape(-1, 1), params.dim, axis=1)
#     ys = params.A * np.sin(params.w * repeat_xs + params.phi) + noise
#     print('Y[0][0]:', ys[0, 0])
#     print('Y[-1][0]:', ys[-1, 0])
#     print('Y[:20][0]:', ys[:20, 0])
#
#     # Plot component 0.
#     if plot:
#         plt.title(params.title)
#         plt.plot(ys[:, 0])
#         plt.savefig(os.path.join(DATA_DIR, params.name + '_line.png'))
#         plt.show()
#
#     # TODO Save to .pkl file if requested
#     if save:
#         pass
#
#     print('Done Generate:', params.name)
#     return split_train_test(ys, None, split=split)


def main():
    # params = Params(id_=1, seed=0, k=0.001, sigma=0.1,
    #                 m=10, l=200, mu_shift=2, sigma_shift=1)
    # np.random.seed(0)  # TODO
    # params = Params(id_=2, dim=2, k=0.001, sigma=0.1)
    # params = SineParams(id_=3, dim=2, A=1., w=0.005, phi=0., sigma=0.1)
    # train_data, train_labels, test_data, test_labels = generate_sine(params, split=0.7, plot=True, save=False)

    params = Params(id_=1, dim=80, k=0.01, sigma=0)
    train_data, train_labels, test_data, test_labels = _generate_linear(params, plot=True, save=False)

    assert train_labels is None and test_labels is None


if __name__ == '__main__':
    main()
