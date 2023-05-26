import os
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

import utils_common
import utils_mnist
from constants_mnist_pca import *
from ml import dense


def mnist17_01a():
    parent_dir = 'mnist17_01a'
    experiment_dir = parent_dir

    os.makedirs(experiment_dir, exist_ok=True)

    train_x, train_y, test_x, test_y, train_batch_id = utils_mnist.basic_train_test()

    dense_nn = dense.DenseNN(parent_dir=experiment_dir,
                             name=parent_dir,
                             input_dim=784, h1_dim=H1_DIM, h2_dim=H2_DIM,
                             classes=(1, 7), batch_size=BATCH_SIZE,
                             mmap_normalise=False)
    model_dir = dense_nn.model_dir
    print('Model dir:', model_dir)

    pred_cluster_labels_by_epoch = []  # Just for 2 clusters.
    cluster_names = ('Cluster 1', 'Cluster 2')
    cluster_colours = ('blue', 'red')
    mmap_pca_by_epoch = []

    for epoch in range(NUM_EPOCHS):
        current_epoch = len(dense_nn.epoch_mmaps) + 1

        dense_nn.fit(train_x, train_y,
                     validation_data=(test_x, test_y), batch_size=BATCH_SIZE, num_epochs=1, continue_training=True)

        # Model is loaded during fit(). Start from scratch every time.
        assert len(dense_nn.epoch_mmaps) == epoch + 1

        mmap, _, _ = dense_nn.epoch_mmaps[-1]  # mmap mush already be demeaned here.
        assert mmap.shape[0] == NUM_TRAIN_BATCHES

        pca = PCA(n_components=3)
        mmap_pca = pca.fit_transform(mmap)
        mmap_pca_by_epoch.append(mmap_pca)
        explained_var = [round(val, 5) for val in pca.explained_variance_]
        singular_values = [round(val, 5) for val in pca.singular_values_]

        # Try to cluster into 2 clusters.
        kmeans = KMeans(n_clusters=2, random_state=0).fit(mmap_pca)
        pred_cluster_masks = [kmeans.labels_ == i for i in range(2)]
        pred_cluster_labels_by_epoch.append(kmeans.labels_)

        true_cluster_masks = [[True for _ in kmeans.labels_]]  # Just one cluster.

        # Components 1 and 2.
        utils_common.plot_mmap_pca(mmap_pca, current_epoch, pred_cluster_masks, cluster_names, cluster_colours,
                                   explained_var, singular_values, 0, 1, model_dir, is_true=False)
        utils_common.plot_mmap_pca(mmap_pca, current_epoch, true_cluster_masks, ['True'], ['blue'],
                                   explained_var, singular_values, 0, 1, model_dir, is_true=True)

        # Components 1 and 3.
        utils_common.plot_mmap_pca(mmap_pca, current_epoch, pred_cluster_masks, cluster_names, cluster_colours,
                                   explained_var, singular_values, 0, 2, model_dir, is_true=False)
        utils_common.plot_mmap_pca(mmap_pca, current_epoch, true_cluster_masks, ['True'], ['blue'],
                                   explained_var, singular_values, 0, 2, model_dir, is_true=True)

        # Components 2 and 3.
        utils_common.plot_mmap_pca(mmap_pca, current_epoch, pred_cluster_masks, cluster_names, cluster_colours,
                                   explained_var, singular_values, 1, 2, model_dir, is_true=False)
        utils_common.plot_mmap_pca(mmap_pca, current_epoch, true_cluster_masks, ['True'], ['blue'],
                                   explained_var, singular_values, 1, 2, model_dir, is_true=True)

    # Save the final model to perform model-based poisoning later.
    dense_nn.model.save(experiment_dir + '/MNIST_for_poison.h5')

    # Evaluate quality of the final model.
    with open(experiment_dir + '/eval.txt', 'w') as f:
        f.write(str(dense_nn.model.evaluate(test_x, test_y)))


def main():
    # mnist17_01a()
    pass


if __name__ == '__main__':
    main()
