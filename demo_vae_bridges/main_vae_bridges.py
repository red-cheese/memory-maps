"""
Experiment 01

Unsupervised learning with VAE (no anomalies).

Data parameters:
k: trend angle; varies from 0 to 1 with step of 0.01
sigma: noise; varies from 0 to 1 with step of 0.1
"""


SEED = 1
import numpy as np
np.random.seed(SEED)


import matplotlib.pyplot as plt
import numpy as np
import os
import quant_utils
import synthetic_data
from ml_bridges import vae as vae_
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


# INPUT_DIM = 10
INPUT_DIM = 80
K_MIN = 0.01
K_MAX = 0.02
# NUM_K = 20
NUM_K = 1
SIGMA_MIN = 0.0001
SIGMA_MAX = 1
OUT_DIR = './synthetic_results'
MC_SAMPLES = 1


def _plot_loss_history(history, params):
    assert len(history.all_losses) == vae_.EPOCHS

    # TODO Fix this - construct colours properly
    colours = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'w'][:vae_.EPOCHS]

    for (epoch, batch_losses), colour in zip(enumerate(history.all_losses), colours):
        batch_losses = np.array(batch_losses)
        batch_losses[np.isnan(batch_losses)] = 0.
        plt.plot(batch_losses, color=colour, label='epoch {}'.format(epoch + 1))

    plt.title('Training losses on batches\nk: {}'.format(params.k))
    plt.xlabel('Batch ID')
    plt.ylabel('Loss Value')
    plt.legend(loc='upper right')
    plt.savefig(os.path.join(params.dir, 'loss_history_{}.png'.format(params.name)))
    plt.gcf().clear()

    # Now plot positive log losses.
    # TODO Think of a better metric?
    # TODO Also possibly loos at validation/testing losses?
    for (epoch, batch_losses), colour in zip(enumerate(history.all_losses), colours):
        batch_losses = np.array(batch_losses)
        batch_losses[np.isnan(batch_losses)] = 0.
        batch_losses[batch_losses <= 0] = 1e-8
        batch_losses = np.log(batch_losses)
        plt.plot(batch_losses, color=colour, label='epoch {}'.format(epoch + 1))

    plt.title('Training log losses on batches\nk: {}'.format(params.k))
    plt.xlabel('Batch ID')
    plt.ylabel('Log Loss Value')
    plt.legend(loc='upper right')
    plt.savefig(os.path.join(params.dir, 'logloss_history_{}.png'.format(params.name)))
    plt.gcf().clear()


def _run_exp(params, shuffle_training_data, plot=False):
    import bridge_data_utils
    train_data, train_labels = bridge_data_utils.get_data(1, diff=50)

    print('Clip train and test data to conform with VAE')
    print('Original training set size:', len(train_data))
    train_data = train_data[:-(len(train_data) % vae_.BATCH_SIZE), :]
    print('Clipped training set size:', len(train_data))
    #print('Original testing set size:', len(test_data))
    #test_data = test_data[:-(len(test_data) % vae_.BATCH_SIZE), :]
    #print('Clipped testing set size:', len(test_data))
    #print()
    train_data_idx = np.arange(train_data.shape[0])
    #test_data_idx = np.arange(test_data.shape[0]) + len(train_data_idx)

    data_mu = np.mean(train_data, axis=0)
    data_sigma = np.std(train_data, axis=0)
    train_data = (train_data - data_mu) / data_sigma

    vae = vae_.DenseVAE(input_dim=80)
    history = vae.fit(train_data, shuffle=shuffle_training_data)
    # if plot:
    #     _plot_loss_history(history, params)

    # pca = PCA(n_components=vae_.LATENT_DIM)
    # scaler = StandardScaler()
    # train_data_scaled = scaler.fit_transform(train_data)
    # pca.fit(train_data_scaled)

    # Predict Dense VAE.
    # NB: Testing data are never shuffled!
    # x_mus_test, x_logsigmas_test = vae.predict(test_data)
    # recprobas = quant_utils.rec_proba(test_data, x_mus_test, x_logsigmas_test)
    # recprobas_mean = np.mean(recprobas)
    # recprobas_std = np.std(recprobas)
    # z_mus_test, z_logsigmas_test = vae._encoder.predict(test_data, batch_size=vae_.BATCH_SIZE)
    # z_mus_mean = np.mean(z_mus_test, axis=0)
    # z_logsigmas_mean = np.mean(z_logsigmas_test, axis=0)

    # Predict PCA.
    # test_data_scaled = scaler.transform(test_data)
    # pca_z_test = pca.transform(test_data_scaled)

    # if plot:
    #     # Plot Dense VAE reconstruction (component 0).
    #     plt.title('Reconstruction (component 0)\nk: {}'.format(params.k))
    #     x_mus_train, _ = vae.predict(train_data)
    #     plt.plot(train_data_idx, x_mus_train[:, 0], color='orange', label='Reconstruction (train)')
    #     plt.plot(test_data_idx, x_mus_test[:, 0], color='red', label='Reconstruction (test)')
    #     plt.plot(train_data_idx, train_data[:, 0], color='green', label='True data (train)')
    #     plt.plot(test_data_idx, test_data[:, 0], color='blue', label='True data (test)')
    #     plt.xlabel('Entry ID')
    #     plt.ylabel('Value')
    #     plt.legend(loc='upper left')
    #     plt.savefig(os.path.join(params.dir, 'dvae_reconst_{}.png'.format(params.name)))
    #     plt.gcf().clear()
    #
    #     # Plot PCA reconstruction (component 0).
    #     plt.title('Reconstruction (component 0)\nk: {}'.format(params.k))
    #     pca_z_train = pca.transform(train_data_scaled)
    #     pca_x_train = pca.inverse_transform(pca_z_train)
    #     pca_x_test = pca.inverse_transform(pca_z_test)
    #     plt.plot(train_data_idx, pca_x_train[:, 0], color='orange', label='Reconstruction (train)')
    #     plt.plot(train_data_idx, train_data_scaled[:, 0], color='green', label='True data (train)')
    #     plt.plot(test_data_idx, test_data_scaled[:, 0], color='blue', label='True data (test)')
    #     plt.plot(test_data_idx, pca_x_test[:, 0], color='red', label='Reconstruction (test)')
    #     plt.xlabel('Entry ID')
    #     plt.ylabel('Value')
    #     plt.legend(loc='upper left')
    #     plt.savefig(os.path.join(params.dir, 'pca_reconst_0_{}.png'.format(params.name)))
    #     plt.gcf().clear()
    #
    #     # Plot PCA reconstruction (component 1).
    #     plt.title('Reconstruction (component 1)\nk: {}'.format(params.k))
    #     pca_z_train = pca.transform(train_data_scaled)
    #     pca_x_train = pca.inverse_transform(pca_z_train)
    #     pca_x_test = pca.inverse_transform(pca_z_test)
    #     plt.plot(train_data_idx, pca_x_train[:, 1], color='orange', label='Reconstruction (train)')
    #     plt.plot(train_data_idx, train_data_scaled[:, 1], color='green', label='True data (train)')
    #     plt.plot(test_data_idx, test_data_scaled[:, 1], color='blue', label='True data (test)')
    #     plt.plot(test_data_idx, pca_x_test[:, 1], color='red', label='Reconstruction (test)')
    #     plt.xlabel('Entry ID')
    #     plt.ylabel('Value')
    #     plt.legend(loc='upper left')
    #     plt.savefig(os.path.join(params.dir, 'pca_reconst_1_{}.png'.format(params.name)))
    #     plt.gcf().clear()
    #
    #     # Plot latent spaces.
    #     plt.title('Encodings: PCA vs VAE\nk: {}'.format(params.k))
    #     plt.scatter(pca_x_train[:, 0], pca_z_train[:, 1], color='green', label='PCA (train)')
    #     z_mus_train, _ = vae._encoder.predict(train_data, batch_size=vae_.BATCH_SIZE)
    #     plt.scatter(z_mus_train[:, 0], z_mus_train[:, 1], color='blue', label='Dense VAE (train)')
    #     plt.scatter(pca_z_test[:, 0], pca_z_test[:, 1], color='orange', label='PCA (test)')
    #     plt.scatter(z_mus_test[:, 0], z_mus_test[:, 1], color='red', label='Dense VAE (test)')
    #     plt.xlabel('z[0]')
    #     plt.ylabel('z[1]')
    #     plt.legend(loc='upper left')
    #     plt.savefig(os.path.join(params.dir, 'z_pca_dvae_{}.png'.format(params.name)))
    #     plt.gcf().clear()

    # return (
    #     recprobas_mean, recprobas_std,  # Across all data points.
    #     z_mus_mean, z_logsigmas_mean,  # Across all data points.
    # )


def _experiment_010(results_dir, shuffle_training_data):
    """Sigma (noise) is kept constant and very small; k (trend angle) is varying."""

    sigma = 0.

    results_dir += '_noise{}_steps{}'.format(sigma, NUM_K)
    if os.path.isdir(results_dir):
        raise ValueError('Experiment results directory already exists:', results_dir)
    os.mkdir(results_dir)

    print('====================================================================')
    print('Experiment 010: small constant sigma; varying k')
    print('***** ***** ***** ***** *****')
    print('Sigma:', sigma)

    # Monitor the following on the testing set:
    #   - Recproba (avg)
    #   - Loss - TODO
    #   - Latent space (Z) mu and sigma (avg)
    test_recproba_means = []
    test_recproba_stds = []
    test_z_mu_means = []
    test_z_logsigma_means = []

    ks = np.linspace(start=K_MIN, stop=K_MAX, endpoint=True, num=NUM_K)
    print('Start varying K')
    for id_, k in enumerate(ks):
        print('***** ***** ***** ***** *****')
        print('Step:', id_, '; k:', k)
        mc_recbroba_means = []
        mc_recproba_stds = []
        mc_z_mu_means = []
        mc_z_logsigma_means = []

        for i in range(MC_SAMPLES):
            params = synthetic_data.Params(id_=id_, dim=INPUT_DIM, k=k, sigma=sigma, results_dir=results_dir)
            (recprobas_mean, recprobas_std, z_mu_mean, z_logsigma_mean,) = _run_exp(
                params, shuffle_training_data, plot=i == MC_SAMPLES - 1)
            # mc_recbroba_means.append(recprobas_mean)
            # mc_recproba_stds.append(recprobas_std)
            # mc_z_mu_means.append(z_mu_mean)
            # mc_z_logsigma_means.append(z_logsigma_mean)

        # test_recproba_means.append(np.mean(mc_recbroba_means))
        # test_recproba_stds.append(np.mean(mc_recproba_stds))
        # test_z_mu_means.append(np.mean(mc_z_mu_means, axis=0))
        # test_z_logsigma_means.append(np.mean(mc_z_logsigma_means, axis=0))
        # print('***** ***** ***** ***** *****')
        # print()

    # test_recproba_means = np.asarray(test_recproba_means)
    # test_recproba_stds = np.asarray(test_recproba_stds)
    # print('Test recproba sample means:', test_recproba_means)
    # print('Test recproba sample stds (biased):', test_recproba_stds)
    # test_recproba_means = np.clip(test_recproba_means, -100, 100)
    # test_recproba_stds = np.clip(test_recproba_stds, -100, 100)
    # plt.title('Reconstruction probabilities on the test set (mean)')
    # plt.xlabel('Trend slope (k)')
    # plt.ylabel('Recproba value (mean)')
    # plt.plot(ks, test_recproba_means, '-o', color='blue', alpha=0.7, linewidth=2)
    # plt.fill_between(ks, test_recproba_means - test_recproba_stds, test_recproba_means + test_recproba_stds,
    #                  color='blue', alpha=0.2)
    # plt.savefig(os.path.join(results_dir, 'dvae_test_recproba_means.png'))
    # plt.gcf().clear()

    # print('Test z mu means:', test_z_mu_means)
    # test_z_mu_means = np.asarray(test_z_mu_means)
    # plt.title('Gaussian latent space mus, per-component (mean)')
    # plt.xlabel('Trend slope (k)')
    # plt.ylabel('z_mu[i] (mean)')
    # plt.plot(ks, test_z_mu_means[:, 0], '-o', color='green', label='z_mu[0] (mean)')  # Colour 1 for component 0.
    # plt.plot(ks, test_z_mu_means[:, 1], '-o', color='orange', label='z_mu[1] (mean)')  # Colour 2 for component 1.
    # plt.legend()
    # plt.savefig(os.path.join(results_dir, 'dvae_test_z_mu_means.png'))
    # plt.gcf().clear()

    # print('Test z logsigma means:', test_z_logsigma_means)
    # test_z_logsigma_means = np.asarray(test_z_logsigma_means)
    # plt.title('Gaussian latent space logsigmas, per-component (mean)')
    # plt.xlabel('Trend slope (k)')
    # plt.ylabel('z_logsigma[i] (mean)')
    # plt.plot(ks, test_z_logsigma_means[:, 0], '-o', color='green', label='z_logsigma[0] (mean)')
    # plt.plot(ks, test_z_logsigma_means[:, 1], '-o', color='orange', label='z_logsigma[1] (mean)')
    # plt.legend()
    # plt.savefig(os.path.join(results_dir, 'dvae_test_z_logsigma_means.png'))
    # plt.gcf().clear()

    print('====================================================================')
    print('Experiment 010 completed')
    print('====================================================================')
    print()


def main():
    shuffle = False
    results_dir = os.path.join(OUT_DIR, 'full_bridge1_e100_seed{}_mmap_losshistory_shuffle{}'.format(SEED, shuffle))
    _experiment_010(results_dir, shuffle)


if __name__ == '__main__':
    main()
