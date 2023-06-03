

import quant_utils
from ml_vl import loss_history

from keras import backend as K
from keras import Model
from keras import initializers
from keras.layers import Dense, Input, Lambda
from keras.optimizers.legacy.adam import Adam
import os
import numpy as np
import pickle


BATCH_SIZE = 256
# EPOCHS = 50
EPOCHS = 4
LATENT_DIM = 2
STDDEV_INIT = 0.01
KERNEL_INIT = initializers.RandomNormal(stddev=STDDEV_INIT)
KERNEL_REG = 'l2'

# MIN_PREDICTED_LOGSIGMA = np.log(0.0001)  # For better stability of recprobas.


class VAE:
    """
    Variational Auto-Encoder (Kingma & Welling, 2013).
    """

    _MODEL_NAME = 'vae_epochs{e}_batch{b}_ldim{ld}'.format(e=EPOCHS, b=BATCH_SIZE, ld=LATENT_DIM)
    _MODEL_FILE_PATTERN = './models/{}.model'

    def __init__(self, input_dim, suffix=None,
                 shuffle_training_data=True, save=True, mmap=False):
        self._name = self._MODEL_NAME
        if suffix:
            self._name += '_{}'.format(suffix)
        self._file = self._MODEL_FILE_PATTERN.format(self._name)

        self._input_dim = input_dim
        self._encoder, self._generator, self._vae = self._build_model()

        # TODO Remove this when separating the bridges repository
        self._shuffle_training_data = shuffle_training_data
        self._save = save
        self._mmap = mmap

    def _build_model(self):
        # Encoding.
        x = Input(shape=(self._input_dim,))
        h = Dense(32, activation='relu', kernel_initializer=KERNEL_INIT, kernel_regularizer=KERNEL_REG)(x)
        z_mean = Dense(LATENT_DIM, kernel_initializer=KERNEL_INIT, kernel_regularizer=KERNEL_REG)(h)
        z_log_sigma = Dense(LATENT_DIM, kernel_initializer=KERNEL_INIT, kernel_regularizer=KERNEL_REG)(h)

        def sampling(args):
            z_mean, z_log_sigma = args
            # 1 sample per data point is enough.
            epsilon = K.random_normal(shape=(BATCH_SIZE, LATENT_DIM),
                                      mean=0., stddev=1.)
            return z_mean + K.exp(z_log_sigma) * epsilon

        # Sampling from latent space.
        z = Lambda(sampling, output_shape=(LATENT_DIM,))([z_mean, z_log_sigma])

        # Decoding samples into a prediction (with incorporated uncertainty).
        decoder_h = Dense(32, activation='relu', kernel_initializer=KERNEL_INIT, kernel_regularizer=KERNEL_REG)
        decoder_mean = Dense(self._input_dim, kernel_initializer=KERNEL_INIT, kernel_regularizer=KERNEL_REG)
        # Assume diagonal covariance matrix for now.
        decoder_log_sigma_2 = Dense(self._input_dim, kernel_initializer=KERNEL_INIT, kernel_regularizer=KERNEL_REG)
        h_decoded = decoder_h(z)
        x_decoded_mean = decoder_mean(h_decoded)
        x_decoded_log_sigma_2 = decoder_log_sigma_2(h_decoded)

        # End-to-end autoencoder
        vae = Model(x, [x_decoded_mean, x_decoded_log_sigma_2])

        # Encoder, from inputs to latent space.
        encoder = Model(x, [z_mean, z_log_sigma])

        # Generator, from latent space to reconstructed inputs.
        decoder_input = Input(shape=(LATENT_DIM,))
        _h_decoded = decoder_h(decoder_input)
        _x_decoded_mean = decoder_mean(_h_decoded)
        _x_decoded_log_sigma_2 = decoder_log_sigma_2(_h_decoded)  # Might be worth clipping it.
        generator = Model(decoder_input, [_x_decoded_mean, _x_decoded_log_sigma_2])

        reconstruction_loss = -K.sum(
            # x_decoded_log_sigma_2 - matrix of shape (batch_size, input dim = 80)
            -(0.5 * np.log(2 * np.pi) + 0.5 * x_decoded_log_sigma_2)
            # x, x_decoded_mean - matrices of shape (batch_size, input dim = 80)
            - 0.5 * (K.square(x - x_decoded_mean) / K.exp(x_decoded_log_sigma_2)),
            axis=1
        )
        kl_loss = 1 + z_log_sigma - K.square(z_mean) - K.exp(z_log_sigma)
        kl_loss = K.sum(kl_loss, axis=-1)
        kl_loss *= -0.5
        # Mean over batch elements.
        vae_loss = K.mean(reconstruction_loss + kl_loss)

        vae.add_loss(vae_loss)
        vae.compile(optimizer=Adam(learning_rate=1e-4))

        return encoder, generator, vae

    def fit(self, train_data):
        print('*** VAE: Training ***')
        print()

        if os.path.isfile(self._file):
            print('Loading model from', self._file)
            self._vae.load_weights(self._file)
            history = None

        else:
            if self._shuffle_training_data:
                idx = np.arange(train_data.shape[0])
                np.random.shuffle(idx)
                train_data = train_data[idx, :]

            print('Fitting, shuffle = ', self._shuffle_training_data)
            # history = loss_history.LossHistory()
            if self._mmap:
                mmap = loss_history.MemoryMap(train_data, BATCH_SIZE, self)
                # callbacks = [mmap, history]
                callbacks = [mmap]
                # callbacks = [history]
            else:
                callbacks = []
            self._vae.fit(train_data,
                          #nb_epoch=EPOCHS,
                          epochs=EPOCHS,
                          batch_size=BATCH_SIZE,
                          shuffle=self._shuffle_training_data,
                          # verbose=1,
                          verbose=0,
                          callbacks=callbacks)

            if self._save:
                print('Saving model to', self._file)
                self._vae.save_weights(self._file)

        print()
        print('*** VAE: Training completed ***')
        print()

        #return history

    def predict(self, test_data):
        output_mus, output_logsigmas = self._vae.predict(test_data, batch_size=BATCH_SIZE, verbose=0)
        return output_mus, output_logsigmas


class VAEClassifier(VAE):

    def __init__(self, input_dim, suffix=None):
        super(VAEClassifier, self).__init__(input_dim=input_dim, suffix=suffix)
        self._recproba_threshold = -130  # Set empirically, based on dataset 1.

    def fit(self, train_data, dump_latent=False, dump_latent_true_labels=None):
        print('*** VAEClassifier: Training the model ***')
        super(VAEClassifier, self).fit(train_data)
        print('*** VAEClassifier: Training completed ***')
        print()

        output_mus, output_logsigmas = self._vae.predict(train_data, batch_size=BATCH_SIZE)
        # output_logsigmas = np.clip(output_logsigmas, a_max=None, a_min=MIN_PREDICTED_LOGSIGMA)
        recprobas = quant_utils.rec_proba(train_data, output_mus, output_logsigmas)
        recproba_mean = np.mean(recprobas)
        recproba_std = np.std(recprobas)
        print('* Rec proba mean:', recproba_mean)
        print('* Rec proba std:', recproba_std)
        print('* VAE rec proba threshold:', self._recproba_threshold)

        # Save to plot latent space, marked event/no event.
        if dump_latent:
            assert dump_latent_true_labels is not None
            plot_z_data = train_data
            z_mus, z_logsigmas = self._encoder.predict(plot_z_data, batch_size=BATCH_SIZE)
            print('Dumping VAE latent space', dump_latent_true_labels.shape)
            with open('figure7.pkl', 'wb') as f:
                pickle.dump((z_mus, z_logsigmas, dump_latent_true_labels), f)

    def predict(self, test_data):
        mus, logsigmas = super(VAEClassifier, self).predict(test_data)
        rec_probas = quant_utils.rec_proba(test_data, mus, logsigmas)
        preds = np.where(rec_probas < self._recproba_threshold, 1., 0.)
        return preds
