

from ml_bridges import rec_proba
from ml_bridges import loss_history

from keras import backend as K
from keras import Model
from keras import initializers
from keras.layers import Dense, Input, Lambda, LSTM, RepeatVector, Reshape
from keras.optimizers.legacy.adam import Adam
import os
import numpy as np
import pickle


BATCH_SIZE = 64
EPOCHS = 4
LATENT_DIM = 2
STDDEV_INIT = 0.01
KERNEL_INIT = initializers.RandomNormal(stddev=STDDEV_INIT)
KERNEL_REG = 'l2'


class _VAEBase:
    """
    Variational Auto-Encoder (Kingma and Welling, 2013).
    """

    _NAME = None
    _MODEL_NAME = None
    _MODEL_FILE_PATTERN = './models/{}.model'

    def __init__(self, input_dim, suffix=None):
        self._name = self._MODEL_NAME
        if suffix:
            self._name += '_{}'.format(suffix)
        self._file = self._MODEL_FILE_PATTERN.format(self._name)
        self._input_dim = input_dim
        self._vae, self._encoder = self._build_model()

    def _build_model(self):
        raise NotImplementedError

    def fit(self, train_data, shuffle=True):
        print('*** VAE: Training ***')

        if os.path.isfile(self._file):
            print('Loading model from', self._file)
            self._vae.load_weights(self._file)
        else:
            print('Fitting')
            mmap = loss_history.MemoryMap(train_data, BATCH_SIZE, self)
            callbacks = [mmap]
            self._vae.fit(train_data,
                          epochs=EPOCHS,
                          batch_size=BATCH_SIZE,
                          shuffle=shuffle,
                          verbose=0,
                          callbacks=callbacks)
            print('Saving model to', self._file)
            self._vae.save_weights(self._file)

        print('*** VAE: Training completed ***')
        print()

    def predict(self, test_data, batch_size=BATCH_SIZE):
        output_mus, output_logsigmas = self._vae.predict(test_data, batch_size=batch_size, verbose=0)
        return output_mus, output_logsigmas


class DenseVAE(_VAEBase):
    """
    Variational Auto-Encoder with Dense networks as its encoder and decoder.
    """

    _NAME = 'dense'
    _MODEL_NAME = 'vae_epochs{e}_batch{b}_ldim{ld}'.format(e=EPOCHS, b=BATCH_SIZE, ld=LATENT_DIM)

    def _build_model(self):
        # Encoding.
        x = Input(shape=(self._input_dim,))
        encoder_h = Dense(40, activation='relu', kernel_initializer=KERNEL_INIT, kernel_regularizer=KERNEL_REG)(x)
        z_mean = Dense(LATENT_DIM, kernel_initializer=KERNEL_INIT, kernel_regularizer=KERNEL_REG)(encoder_h)
        z_log_sigma = Dense(LATENT_DIM, kernel_initializer=KERNEL_INIT, kernel_regularizer=KERNEL_REG)(encoder_h)

        def sampling(args):
            z_mean, z_log_sigma = args
            # 1 sample per 1 data point.
            epsilon = K.random_normal(shape=(BATCH_SIZE, LATENT_DIM),
                                      mean=0., stddev=1.)
            return z_mean + K.exp(z_log_sigma) * epsilon

        # Sampling from latent space.
        z = Lambda(sampling, output_shape=(LATENT_DIM,))([z_mean, z_log_sigma])

        # Decoding samples.
        decoder_h = Dense(40, activation='relu', kernel_initializer=KERNEL_INIT, kernel_regularizer=KERNEL_REG)(z)
        x_decoded_mean = Dense(self._input_dim, kernel_initializer=KERNEL_INIT, kernel_regularizer=KERNEL_REG)(decoder_h)
        # Assume diagonal predicted covariance matrix.
        x_decoded_log_sigma_2 = Dense(self._input_dim, kernel_initializer=KERNEL_INIT, kernel_regularizer=KERNEL_REG)(decoder_h)

        # End-to-end VAE.
        vae = Model(x, [x_decoded_mean, x_decoded_log_sigma_2])
        # Save the encoder part separately as we will need it later.
        encoder = Model(x, [z_mean, z_log_sigma])

        # Loss: -ELBO.
        reconstruction_loss = -K.sum(
            # x_decoded_log_sigma_2 - matrix of shape (batch_size, input dim).
            -(0.5 * np.log(2 * np.pi) + 0.5 * x_decoded_log_sigma_2)
            # x, x_decoded_mean - matrices of shape (batch_size, input dim).
            - 0.5 * (K.square(x - x_decoded_mean) / K.exp(x_decoded_log_sigma_2)),
            axis=1
        )
        kl_loss = 1 + z_log_sigma - K.square(z_mean) - K.exp(z_log_sigma)
        kl_loss = K.sum(kl_loss, axis=-1)
        kl_loss *= -0.5
        # Mean over batch elements.
        vae_loss = K.mean(reconstruction_loss + kl_loss)
        vae.add_loss(vae_loss)
        vae.compile(optimizer=Adam(learning_rate=1e-3))

        return vae, encoder


class RNNVAE(_VAEBase):
    """
    Variational Auto-Encoder with RNNs as its encoder and decoder.
    """

    _NAME = 'rnn'
    _MODEL_NAME = 'rnnvae_epochs{e}_batch{b}_ldim{ld}'.format(e=EPOCHS, b=BATCH_SIZE, ld=LATENT_DIM)

    def _build_model(self):
        timesteps = 1
        features = self._input_dim // timesteps

        # Encoding.
        x = Input(shape=(self._input_dim,))
        x_reshaped = Reshape((timesteps, features))(x)
        encoder_h = LSTM(40, return_sequences=True)(x_reshaped)
        z_mean = LSTM(LATENT_DIM, activation=None)(encoder_h)
        z_log_sigma = LSTM(LATENT_DIM, activation=None)(encoder_h)

        def sampling(args):
            z_mean, z_log_sigma = args
            # 1 sample per 1 data point.
            epsilon = K.random_normal(shape=(BATCH_SIZE, LATENT_DIM),
                                      mean=0., stddev=1.)
            return z_mean + K.exp(z_log_sigma) * epsilon

        # Sampling from latent space.
        z = Lambda(sampling, output_shape=(LATENT_DIM,))([z_mean, z_log_sigma])

        # Decoding samples.
        decoder_h = LSTM(40, return_sequences=True)
        # Don't return sequences as the original input did not have the
        # timesteps axis.
        decoder_mean = LSTM(features, activation=None)
        decoder_std = LSTM(features, activation=None)

        h_decoded = RepeatVector(timesteps)(z)
        h_decoded = decoder_h(h_decoded)

        x_decoded_mean = decoder_mean(h_decoded)
        x_decoded_log_sigma_2 = decoder_std(h_decoded)

        # End-to-end VAE.
        vae = Model(x, [x_decoded_mean, x_decoded_log_sigma_2])
        # Save the encoder part separately as we will need it later.
        encoder = Model(x, [z_mean, z_log_sigma])

        # Loss: -ELBO.
        reconstruction_loss = -K.sum(
            # x_decoded_log_sigma_2 - matrix of shape (batch_size, input dim).
            -(0.5 * np.log(2 * np.pi) + 0.5 * x_decoded_log_sigma_2)
            # x, x_decoded_mean - matrices of shape (batch_size, input dim).
            - 0.5 * (K.square(x - x_decoded_mean) / K.exp(x_decoded_log_sigma_2)),
            axis=1
        )
        kl_loss = 1 + z_log_sigma - K.square(z_mean) - K.exp(z_log_sigma)
        kl_loss = K.sum(kl_loss, axis=-1)
        kl_loss *= -0.5
        # Mean over batch elements.
        vae_loss = K.mean(reconstruction_loss + kl_loss)
        vae.add_loss(vae_loss)
        vae.compile(optimizer='adam')

        vae.summary()

        return vae, encoder


class VAEClassifier:
    def __init__(self, vae_class, input_dim, suffix=None,
                 # Set empirically, based on dataset 1.
                 # -130 is for the DenseVAE.
                 recproba_threshold=-130):
        assert vae_class in (DenseVAE, RNNVAE)
        self._vae_class = vae_class
        print('*** VAEClassifier: VAE class: {} ***'.format(vae_class))
        self._vae = vae_class(input_dim=input_dim, suffix=suffix)
        self._recproba_threshold = recproba_threshold

    def fit(self, train_data, shuffle=True,
            dump_latent=False, dump_latent_true_labels=None):
        print('*** VAEClassifier: Training the model ***')
        if self._vae_class is RNNVAE:
            assert not shuffle
        self._vae.fit(train_data, shuffle=shuffle)
        print('*** VAEClassifier: Training completed ***')
        print()

        output_mus, output_logsigmas_2 = self._vae.predict(train_data, batch_size=BATCH_SIZE)
        recprobas = rec_proba.rec_proba(train_data, output_mus, output_logsigmas_2)
        recproba_mean = np.mean(recprobas)
        recproba_std = np.std(recprobas)
        print('* Rec proba mean:', recproba_mean)
        print('* Rec proba std:', recproba_std)
        print('* VAE rec proba threshold:', self._recproba_threshold)

        # Save to plot the latent space, marked event/no event.
        if dump_latent:
            assert dump_latent_true_labels is not None  # Ground truth (manual) labels.
            z_mus, z_logsigmas = self._vae._encoder.predict(train_data, batch_size=BATCH_SIZE)
            print('Dumping VAE latent space')
            with open('figure8_{}.pkl'.format(self._vae._NAME), 'wb') as f:
                pickle.dump((z_mus, z_logsigmas, dump_latent_true_labels), f)

    def predict(self, test_data):
        mus, logsigmas = self._vae.predict(test_data)
        rec_probas = rec_proba.rec_proba(test_data, mus, logsigmas)
        preds = np.where(rec_probas < self._recproba_threshold, 1., 0.)
        return preds
