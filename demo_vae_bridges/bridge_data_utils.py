

import manual_labels
import csv
import matplotlib.figure as figure
import matplotlib.pyplot as plt
import numpy as np
import os
import pickle


IN_DIM = 80
PE_CONST = 0.22  # Photoelastic constant.


# =============================================================================


def _csv_file_path(dataset_id):
    # Wavelength in nanometres.
    return './data/bridge{}.csv'.format(dataset_id)


def _pkl_file_path(dataset_id):
    # In microstrains.
    return './data/bridge{}.pkl'.format(dataset_id)


def _pkl_labelled_file_path(dataset_id):
    return './data/bridge{}_labelled.pkl'.format(dataset_id)


# =============================================================================


def _figure_1():
    """Dataset 1: sum of strains of all sensors around the 1st train event."""

    dataset_id = 1
    pkl_labelled_file = _pkl_labelled_file_path(dataset_id)
    with open(pkl_labelled_file, 'rb') as f:
        data, labels = pickle.load(f)

    l = 79000
    r = 81000
    l_sec = l * 1. / 250
    r_sec = r * 1. / 250

    data = data[l:r, :]
    labels = labels[l:r]
    cdata = np.sum(data, axis=1)
    all_idx = np.arange(cdata.shape[0])
    cdata_no = cdata[labels == 0]
    cdata_yes = cdata[labels == 1]

    plt.xlim(left=l_sec, right=r_sec)
    plt.scatter((all_idx[labels == 0] + l) / 250, cdata_no, color='#aaaaaa', label='No event')
    plt.scatter((all_idx[labels == 1] + l) / 250, cdata_yes, marker='x', c='black', label='Event')
    plt.ticklabel_format(useOffset=False)
    plt.xlabel('Second')
    plt.ylabel('Microstrain')
    plt.legend(loc='lower right')
    plt.savefig('Figure1.png', dpi=300)
    plt.gcf().clear()


def _figure_2():
    """Dataset 3 illustrating the drift in the strains of the 1st sensor."""

    dataset_id = 3
    pkl_file = _pkl_file_path(dataset_id)
    with open(pkl_file, 'rb') as f:
        data = pickle.load(f)

    cdata = data[:, 0]
    seconds = np.arange(data.shape[0]) * 1. / 250

    plt.xlim(right=seconds[-1])
    plt.plot(seconds, cdata, color='black', linestyle=':')
    plt.ticklabel_format(useOffset=False)
    plt.xlabel('Second')
    plt.ylabel('Microstrain')
    plt.savefig('Figure2.png', dpi=300)
    plt.gcf().clear()


def _figure_3():
    """Dataset 3 illustrating the drift in the strains of sensor 34."""

    dataset_id = 3
    pkl_file = _pkl_file_path(dataset_id)
    with open(pkl_file, 'rb') as f:
        data = pickle.load(f)

    cdata = data[:, 33]
    seconds = np.arange(data.shape[0]) * 1. / 250

    plt.xlim(right=seconds[-1])
    plt.plot(seconds, cdata, color='black', linestyle=':')
    plt.ticklabel_format(useOffset=False)
    plt.xlabel('Second')
    plt.ylabel('Microstrain')
    plt.savefig('Figure3.png', dpi=300)
    plt.gcf().clear()


def _figure_4():
    dataset_id = 1
    pkl_file = _pkl_file_path(dataset_id)
    with open(pkl_file, 'rb') as f:
        data = pickle.load(f)

    d1 = data[1:, :]  # Original strain data.
    d2 = _defreq_and_diff(data, None, 1)[0]  # After difference transform, no labels here.
    seconds = (np.arange(d1.shape[0]) + 0) * 1. / 250  # +1, because we start with the 2nd point due to the diff.

    plt.xlim(right=seconds[-1])

    # Plot only sum of sensor values.
    cdata1 = np.sum(d1, axis=1)
    cdata2 = np.sum(d2, axis=1)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax1 = fig.add_subplot(211)
    ax2 = fig.add_subplot(212)
    ax1.set_xlim(right=seconds[-1])
    ax2.set_xlim(right=seconds[-1])

    ax1.title.set_text('\nOriginal')
    ax2.title.set_text('\n\nAfter difference transform')

    plt.ticklabel_format(useOffset=False)
    ax.spines['top'].set_color('none')
    ax.spines['bottom'].set_color('none')
    ax.spines['left'].set_color('none')
    ax.spines['right'].set_color('none')
    ax.tick_params(labelcolor='w', top='off', bottom='off', left='off', right='off')
    ax1.set_xticklabels([])

    ax1.plot(seconds, cdata1, color='black', linestyle=':')
    ax2.plot(seconds, cdata2, color='black', linestyle=':')

    ax.set_xlabel('Second')
    ax.set_ylabel('Microstrain\n')

    plt.savefig('Figure4.png', dpi=300)
    plt.gcf().clear()


def _figure_5():
    dataset_id = 1
    pkl_file = _pkl_file_path(dataset_id)
    with open(pkl_file, 'rb') as f:
        data = pickle.load(f)

    d1 = data[1:, :]
    d2 = _defreq_and_diff(data, None, 1)[0]
    d3 = _defreq_and_diff(data, None, 5)[0]
    d4 = _defreq_and_diff(data, None, 50)[0]

    # Plot only sum of sensor values.
    cdata1 = np.sum(d1, axis=1)
    cdata2 = np.sum(d2, axis=1)
    cdata3 = np.sum(d3, axis=1)
    cdata4 = np.sum(d4, axis=1)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax1 = fig.add_subplot(221)
    ax2 = fig.add_subplot(222)
    ax3 = fig.add_subplot(223)
    ax4 = fig.add_subplot(224)

    ax1.set_xlim(right=d1.shape[0])
    ax2.set_xlim(right=d2.shape[0])
    ax3.set_xlim(right=d3.shape[0])
    ax4.set_xlim(right=d4.shape[0])

    ax1.title.set_text('\nOriginal')
    ax2.title.set_text('\nDiff at 250Hz')
    ax3.title.set_text('\nDiff at 50Hz')
    ax4.title.set_text('\nDiff at 5Hz')

    plt.ticklabel_format(useOffset=False)
    ax.spines['top'].set_color('none')
    ax.spines['bottom'].set_color('none')
    ax.spines['left'].set_color('none')
    ax.spines['right'].set_color('none')
    ax.tick_params(labelcolor='w', top='off', bottom='off', left='off', right='off')
    ax1.set_xticklabels([])
    ax2.set_xticklabels([])
    ax3.set_xticklabels([])
    ax4.set_xticklabels([])

    ax1.plot(cdata1, color='black', linestyle=':')
    ax2.plot(cdata2, color='black', linestyle=':')
    ax3.plot(cdata3, color='black', linestyle=':')
    ax4.plot(cdata4, color='black', linestyle=':')

    ax.set_ylabel('Microstrain\n')

    plt.savefig('Figure5.png', dpi=300)
    plt.gcf().clear()


def _figure_6():
    """Plot of labelled event, zoomed."""

    with open('figure6.pkl', 'rb') as f:
        data_sum, cusum_labels, cusum_pc_1_labels, vae_labels, vae_rnn_labels, true_labels = pickle.load(f)

    l, r = 1570, 1630

    data_sum = data_sum[l:r]
    cusum_labels = cusum_labels[l:r]
    cusum_pc_1_labels = cusum_pc_1_labels[l:r]
    vae_labels = vae_labels[l:r]
    vae_rnn_labels = vae_rnn_labels[l:r]
    true_labels = true_labels[l:r]

    fa = figure.figaspect(0.7)
    fig = plt.figure(figsize=fa)
    ax = fig.add_subplot(111)
    ax1 = fig.add_subplot(511)
    ax2 = fig.add_subplot(512)
    ax3 = fig.add_subplot(513)
    ax4 = fig.add_subplot(514)
    ax5 = fig.add_subplot(515)
    plt.subplots_adjust(hspace=0.43)
    plt.xlim(left=l / 5, right=r / 5)
    ax1.set_xlim(left=l / 5, right=r / 5)
    ax2.set_xlim(left=l / 5, right=r / 5)
    ax3.set_xlim(left=l / 5, right=r / 5)
    ax4.set_xlim(left=l / 5, right=r / 5)
    ax5.set_xlim(left=l / 5, right=r / 5)

    ax1.title.set_text('CUSUM - sum')
    ax2.title.set_text('CUSUM - 1st PC')
    ax3.title.set_text('VAE - Dense')
    ax4.title.set_text('VAE - RNN')
    ax5.title.set_text('Ground truth')

    plt.ticklabel_format(useOffset=False)
    ax.spines['top'].set_color('none')
    ax.spines['bottom'].set_color('none')
    ax.spines['left'].set_color('none')
    ax.spines['right'].set_color('none')
    ax.tick_params(labelcolor='w', top='off', bottom='off', left='off', right='off')
    ax1.set_xticklabels([])
    ax2.set_xticklabels([])
    ax3.set_xticklabels([])
    ax4.set_xticklabels([])

    all_idx = np.arange(data_sum.shape[0])

    # CUSUM (sum) labels.
    ax1.scatter((all_idx[cusum_labels == 0] + l) / 5, data_sum[cusum_labels == 0], edgecolors='black', color='#aaaaaa', label='No event')
    ax1.scatter((all_idx[cusum_labels == 1] + l) / 5, data_sum[cusum_labels == 1], marker='x', c='black', label='Event')
    # CUSUM (1st PC) labels.
    ax2.scatter((all_idx[cusum_pc_1_labels == 0] + l) / 5, data_sum[cusum_pc_1_labels == 0], edgecolors='black', color='#aaaaaa', label='No event')
    ax2.scatter((all_idx[cusum_pc_1_labels == 1] + l) / 5, data_sum[cusum_pc_1_labels == 1], marker='x', c='black', label='Event')
    # VAE Dense labels.
    ax3.scatter((all_idx[vae_labels == 0] + l) / 5, data_sum[vae_labels == 0], edgecolors='black', color='#aaaaaa', label='No event')
    ax3.scatter((all_idx[vae_labels == 1] + l) / 5, data_sum[vae_labels == 1], marker='x', c='black', label='Event')
    # VAE RNN labels.
    ax4.scatter((all_idx[vae_rnn_labels == 0] + l) / 5, data_sum[vae_rnn_labels == 0],
                edgecolors='black', color='#aaaaaa', label='No event')
    ax4.scatter((all_idx[vae_rnn_labels == 1] + l) / 5, data_sum[vae_rnn_labels == 1],
                marker='x', c='black', label='Event')
    # Ground truth.
    ax5.scatter((all_idx[true_labels == 0] + l) / 5, data_sum[true_labels == 0], edgecolors='black', color='#aaaaaa', label='No event')
    ax5.scatter((all_idx[true_labels == 1] + l) / 5, data_sum[true_labels == 1], marker='x', c='black', label='Event')

    plt.legend(loc='lower right', framealpha=1., edgecolor='black')

    ax.set_xlabel('Second')
    ax.set_ylabel('Microstrain\n')

    plt.savefig('Figure6.png', dpi=300)
    plt.gcf().clear()


def _figure_7():
    fa = figure.figaspect(0.7)
    fig = plt.figure(figsize=fa)
    # fig = plt.figure()
    ax = fig.add_subplot(111)
    ax1 = fig.add_subplot(311)
    ax2 = fig.add_subplot(312)
    ax3 = fig.add_subplot(313)
    plt.subplots_adjust(hspace=0.43)

    ax1.title.set_text('Dataset 1')
    ax2.title.set_text('Dataset 2')
    ax3.title.set_text('Dataset 3')

    pkl_file = _pkl_file_path(1)
    with open(pkl_file, 'rb') as f:
        data1 = pickle.load(f)
        cdata1 = np.sum(data1, axis=1)
    pkl_file = _pkl_file_path(2)
    with open(pkl_file, 'rb') as f:
        data2 = pickle.load(f)
        cdata2 = np.sum(data2, axis=1)
    pkl_file = _pkl_file_path(3)
    with open(pkl_file, 'rb') as f:
        data3 = pickle.load(f)
        cdata3 = np.sum(data3, axis=1)

    min_len = min(cdata1.shape[0], cdata2.shape[0], cdata3.shape[0])
    cdata1, cdata2, cdata3 = cdata1[:min_len], cdata2[:min_len], cdata3[:min_len]

    seconds = np.arange(cdata1.shape[0]) * 1. / 250

    ax1.set_xticklabels([])
    ax2.set_xticklabels([])

    plt.ticklabel_format(useOffset=False)
    ax.spines['top'].set_color('none')
    ax.spines['bottom'].set_color('none')
    ax.spines['left'].set_color('none')
    ax.spines['right'].set_color('none')
    ax.tick_params(labelcolor='w', top='off', bottom='off', left='off', right='off')

    ax1.plot(seconds, cdata1, color='black', linestyle=':')
    ax2.plot(seconds, cdata2, color='black', linestyle=':')
    ax3.plot(seconds, cdata3, color='black', linestyle=':')

    ax.set_xlabel('Second')
    ax.set_ylabel('Microstrain\n')
    plt.savefig('Figure7.png', dpi=300)
    plt.gcf().clear()


def _figure_8():
    for name in ('rnn', 'dense'):
        with open('figure8_{}.pkl'.format(name), 'rb') as f:
            z_mus, z_logsigmas, labels = pickle.load(f)
            labels = labels.astype(np.bool)

            plt.rcParams['grid.color'] = 'black'
            plt.rcParams['grid.linewidth'] = 0.5
            plt.rcParams['grid.linestyle'] = ':'
            plt.ticklabel_format(useOffset=False)
            plt.scatter(z_mus[~labels, 0], z_mus[~labels, 1], c='#aaaaaa', edgecolors='black', label='No event')
            plt.scatter(z_mus[labels, 0], z_mus[labels, 1], marker='x', c='black', label='Event')
            plt.xlabel('z[0]')
            plt.ylabel('z[1]')
            lg = plt.legend(loc='lower left', framealpha=1., edgecolor='black')
            lg.draw_frame(True)
            plt.savefig('Figure8_{}.png'.format(name), dpi=300)
            plt.gcf().clear()


# =============================================================================


def _load_csv(csv_file, labeled=True):
    data = []
    labels = []

    with open(csv_file, 'r') as f:
        reader = csv.reader(f, delimiter=',')
        _ = next(reader)

        for line in reader:
            if labeled:
                data.append(np.array(line[1:-1], dtype=np.float64))
                labels.append(float(line[-1]))
            else:
                data.append(np.array(line[1:], dtype=np.float64))
                labels.append(None)

    data = np.array(data).reshape((len(data), IN_DIM))
    labels = np.array(labels)
    assert len(data) == len(labels)

    return data, labels


def _convert_to_microstrains(dataset_id):
    csv_file = _csv_file_path(dataset_id)
    pkl_file = _pkl_file_path(dataset_id)

    print('Loading data from', csv_file)
    # The labels provided are not precise, so we drop them.
    data, _ = _load_csv(csv_file, labeled=True)

    # The 3rd dataset has 2 broken entries which are not related to train events.
    # In this broken entries, sensor measurements of two sensors are just equal to 0.
    # Here's the hack to fix them.
    if dataset_id == 3:
        data[99534, 78] = data[99533, 78]
        data[239379, 67] = data[239378, 67]

    # Plot for debugging purposes.
    print('Plotting data in wavelength')
    data_sum = np.sum(data, axis=1)
    plt.plot(data_sum)
    plt.savefig('{}_wave_sum.png'.format(csv_file))
    plt.gcf().clear()

    print('Converting data to microstrains')
    data_0 = np.tile(data[0, :], (data.shape[0], 1))
    data_ms = 1 / (1 - PE_CONST) * ((data - data_0) / data_0) * 1e6

    # Plot for debugging purposes.
    print('Plotting data in strains')
    data_ms_sum = np.sum(data_ms, axis=1)
    plt.plot(data_ms_sum)
    plt.savefig('{}_microstrain_sum.png'.format(csv_file))
    plt.gcf().clear()

    print('Dumping data in microstrains')
    with open(pkl_file, 'wb') as f:
        pickle.dump(data_ms, f)


def _label(dataset_id):
    pkl_file = _pkl_file_path(dataset_id)
    pkl_labelled_file = _pkl_labelled_file_path(dataset_id)

    with open(pkl_file, 'rb') as f:
        data = pickle.load(f)

    labels = np.zeros(shape=(data.shape[0],), dtype=np.int32)
    for start, end in manual_labels.LABELS[dataset_id]:
        labels[start:(end + 1)] = 1

    event_count = labels.sum()
    print('Dataset {}: total {}, event {}, no event {}'
          .format(dataset_id, data.shape[0], event_count, data.shape[0] - event_count))

    with open(pkl_labelled_file, 'wb') as f:
        pickle.dump((data, labels), f)


def _defreq_and_diff(data, labels, step):
    # Lower the frequency.
    keep_idx = np.arange(0, data.shape[0], step=step)
    new_data = data[keep_idx]
    new_labels = labels[keep_idx] if labels is not None else None
    # Now diff.
    new_data = new_data[1:, :] - new_data[:-1, :]
    new_labels = new_labels[1:] if new_labels is not None else None
    return new_data, new_labels


# =============================================================================


def get_data(dataset_id,
             # Defreq and diff to detrend: diff=50 would mean defreq from 250Hz to 5Hz.
             diff=None):
    pkl_labelled_file = _pkl_labelled_file_path(dataset_id)
    prefix = 'bridge{dataset_id}_diff={diff}'.format(dataset_id=dataset_id, diff=diff)
    filename = './data/{prefix}_labelled.pkl'.format(prefix=prefix)

    if os.path.isfile(filename):
        print('Load existing dataset from', filename)
        with open(filename, 'rb') as f:
            data, labels = pickle.load(f)

    else:
        with open(pkl_labelled_file, 'rb') as f:
            data, labels = pickle.load(f)
        if diff is not None:
            print('Defreq and diff')
            data, labels = _defreq_and_diff(data, labels, diff)
        print('Dumping data to', filename)
        with open(filename, 'wb') as f:
            pickle.dump((data, labels), f)

    # plt.plot(np.sum(data, axis=1))
    # plt.show()

    return data, labels


def prepare_data():
    _convert_to_microstrains(1)
    _convert_to_microstrains(2)
    _convert_to_microstrains(3)
    _label(1)
    _label(2)
    _label(3)


def plot_all():
    _figure_1()
    _figure_2()
    _figure_3()
    _figure_4()
    _figure_5()
    _figure_6()
    _figure_7()
    _figure_8()


def main():
    prepare_data()
    # plot_all()


if __name__ == '__main__':
    main()
