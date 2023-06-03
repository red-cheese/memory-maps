

import keras
import matplotlib.pyplot as plt
import numpy as np
import pickle
import seaborn as sns


_MAX_LOSS = 200
_MIN_LOSS = -200


# TODO Double check the loss: why z_log_sigma, not z_log_sigma_2?
def _vae_loss_fn(x, x_decoded_mean, x_decoded_log_sigma_2, z_mean, z_log_sigma):
    reconstruction_loss = -np.sum(
        # x_decoded_log_sigma_2 - matrix of shape (batch_size, input dim)
        -(0.5 * np.log(2 * np.pi) + 0.5 * x_decoded_log_sigma_2)
        # x, x_decoded_mean - matrices of shape (batch_size, input dim)
        - 0.5 * (np.square(x - x_decoded_mean) / np.exp(x_decoded_log_sigma_2)),
        axis=1)
    kl_loss = 1 + z_log_sigma - np.square(z_mean) - np.exp(z_log_sigma)
    kl_loss = np.sum(kl_loss, axis=-1)
    kl_loss *= -0.5

    # Mean over all elements of the mini-batch.
    return np.mean(reconstruction_loss + kl_loss)


def _vae_loss_fn_nomean(x, x_decoded_mean, x_decoded_log_sigma_2, z_mean, z_log_sigma):
    reconstruction_loss = -np.sum(
        # x_decoded_log_sigma_2 - matrix of shape (batch_size, input dim)
        -(0.5 * np.log(2 * np.pi) + 0.5 * x_decoded_log_sigma_2)
        # x, x_decoded_mean - matrices of shape (batch_size, input dim)
        - 0.5 * (np.square(x - x_decoded_mean) / np.exp(x_decoded_log_sigma_2)),
        axis=1)
    kl_loss = 1 + z_log_sigma - np.square(z_mean) - np.exp(z_log_sigma)
    kl_loss = np.sum(kl_loss, axis=-1)
    kl_loss *= -0.5

    return reconstruction_loss + kl_loss


def _compute_vae_loss(batch, vae):
    x_decoded_mean, x_decoded_log_sigma_2 = vae.predict(batch)
    # As the uncertainty only comes from sampling, z_mean and z_log_sigma
    # must always stay same for one given input.
    z_mean, z_log_sigma = vae._encoder.predict(batch, batch_size=batch.shape[0], verbose=0)
    # return _vae_loss_fn(batch, x_decoded_mean, x_decoded_log_sigma_2, z_mean, z_log_sigma)
    return _vae_loss_fn_nomean(batch, x_decoded_mean, x_decoded_log_sigma_2, z_mean, z_log_sigma)


class MemoryMap(keras.callbacks.Callback):
    """
    Memory map of batch losses within one epoch.
    """

    def __init__(self, all_data, batch_size, vae):
        assert all_data.shape[0] % batch_size == 0

        num_batches = int(all_data.shape[0] / batch_size)
        self.K = num_batches
        self.batch_size = batch_size

        self.vae = vae
        # self.loss_fn = loss_fn
        self.all_data = all_data

        # Rows: batches (fixed)
        # Columns: losses after each gradient step
        # self.mmap[i, j] = loss on batch i after gradient update on batch j has been done
        self.mmap = np.zeros(shape=(self.K, self.K))

        self.cur_batch_id = -1  # Will start with the mini-batch 0.
        self.cur_epoch_id = -1

        # self.batches_dict = {}
        # for i in range(self.K):
        #     batch_start = i * self.batch_size
        #     batch_end = batch_start + self.batch_size
        #     self.batches_dict[i] = self.all_data[batch_start:batch_end, ...]

    def on_batch_begin(self, batch, logs=None):
        self.cur_batch_id += 1

        #if self.cur_batch_id % 5 == 0:
        print('Starting batch', self.cur_batch_id)

    def on_batch_end(self, batch, logs=None):
        loss = _compute_vae_loss(self.all_data, self.vae)
        if np.any(np.isnan(loss)):
            raise RuntimeError('Divergence')
        #print('Loss shape:', loss.shape)
        for i in range(self.K):
            batch_start = i * self.batch_size
            self.mmap[i, self.cur_batch_id] = np.nanmean(loss[batch_start:(batch_start + self.batch_size)])

        # for i in range(self.K):
        #     # batch_start = i * self.batch_size
        #     # batch_end = batch_start + self.batch_size
        #     # b = self.all_data[batch_start:batch_end, ...]
        #     #loss = _compute_vae_loss(b, self.vae)
        #     #loss = self.vae._vae.evaluate(b, batch_size=self.batch_size, verbose=0)
        #     self.mmap[i, self.cur_batch_id] = _vae_.evaluate(self.batches_dict[i], batch_size=self.batch_size, verbose=0)

        # if self.cur_batch_id % 5 == 0:
        print('Ending batch', self.cur_batch_id)

    def on_epoch_begin(self, epoch, logs=None):
        self.cur_epoch_id += 1
        print('Starting epoch', self.cur_epoch_id)

    def on_epoch_end(self, epoch, logs=None):
        #import pdb; pdb.set_trace()
        mmap = self.mmap
        with open(f'epoch{self.cur_epoch_id + 1}.pkl', 'wb') as f:
            pickle.dump(mmap, f)

        isnan = np.isnan(mmap)
        if isnan.all():
            mmap[:, :] = _MAX_LOSS
        else:
            nanmax, nanmin = min(np.nanmax(mmap), _MAX_LOSS), max(np.nanmin(mmap), _MIN_LOSS)
            mmap[isnan] = nanmax
            mmap = np.clip(mmap, nanmin, nanmax)

        sns.heatmap(mmap, xticklabels=self.K // 10, yticklabels=self.K // 10, cmap="YlGnBu")
        plt.title('Memory map (loss): epoch {}'.format(self.cur_epoch_id + 1))
        plt.ylabel('Mini-batch')
        plt.xlabel('Training step')
        plt.savefig('epoch{}.png'.format(self.cur_epoch_id + 1))  # TODO Full path
        plt.gcf().clear()

        print('Ending epoch', self.cur_epoch_id)
        self.mmap = np.zeros(shape=(self.K, self.K))
        self.cur_batch_id = -1


class LossHistory(keras.callbacks.Callback):
    def __init__(self):
        self.all_losses = None
        self.batch_losses = None

    def on_train_begin(self, logs={}):
        self.all_losses = []
        self.batch_losses = []

    def on_epoch_begin(self, epoch, logs=None):
        self.batch_losses = []

    def on_batch_end(self, batch, logs={}):
        self.batch_losses.append(logs.get('loss'))

    def on_epoch_end(self, epoch, logs=None):
        self.all_losses.append(self.batch_losses)
        self.batch_losses = []


def plot_mmap(mmap_dir, epoch, max_loss=_MAX_LOSS, min_loss=_MIN_LOSS):
    mmap_file = f'{mmap_dir}/epoch{epoch}.pkl'
    with open(mmap_file, 'rb') as f:
        mmap = pickle.load(f)

    isnan = np.isnan(mmap)
    if isnan.all():
        mmap[:, :] = max_loss
    else:
        nanmax, nanmin = min(np.nanmax(mmap), max_loss), max(np.nanmin(mmap), min_loss)
        mmap[isnan] = nanmax
        mmap = np.clip(mmap, nanmin, nanmax)

    sns.heatmap(mmap, xticklabels=mmap.shape[0] // 10, yticklabels=mmap.shape[0] // 10, cmap="YlGnBu")
    plt.title('Memory map (loss): epoch {}'.format(epoch))
    plt.ylabel('Mini-batch')
    plt.xlabel('Training step')
    plt.savefig(f'{mmap_dir}/epoch{epoch}_new.png')
    plt.gcf().clear()


def main():
    mmap_dir = '../save-64-32-noshuffle'
    max_loss = 600
    min_loss = -600

    plot_mmap(mmap_dir, 1, max_loss=500, min_loss=-500)
    plot_mmap(mmap_dir, 2, max_loss=500, min_loss=-500)
    plot_mmap(mmap_dir, 3, max_loss=1000, min_loss=-1000)
    plot_mmap(mmap_dir, 4, max_loss=1000, min_loss=-1000)


if __name__ == '__main__':
    main()
