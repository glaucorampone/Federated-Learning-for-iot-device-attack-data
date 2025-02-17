"""our-federation: A Flower / sklearn app."""

import numpy as np
from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import IidPartitioner
from sklearn.ensemble import RandomForestClassifier

fds = None  # Cache FederatedDataset


def load_data(partition_id: int, num_partitions: int, subset: float = 1.0):
    # Only initialize `FederatedDataset` once
    global fds
    if fds is None:
        partitioner = IidPartitioner(num_partitions=num_partitions)
        fds = FederatedDataset(
            dataset="n3p7un/KitsuneSystemAttackData_osScanDataset",
            partitioners={"train": partitioner},
        )

    dataset = fds.load_partition(partition_id, "train").with_format("numpy")

    subset_size = int(len(dataset) * subset)
    dataset = dataset[:subset_size]

    X, y = dataset.remove_columns('label'), dataset["label"]

    # Split the on edge data: 80% train, 20% test
    X_train, X_test = X[: int(0.8 * len(X))], X[int(0.8 * len(X)) :]
    y_train, y_test = y[: int(0.8 * len(y))], y[int(0.8 * len(y)) :]

    return X_train, X_test, y_train, y_test


def get_model(n_estimators: int, max_depth: int):

    return RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        warm_start=True,
    )


def get_model_params(model):
    params = model.get_params()
    return params


def set_model_params(model, params):
    model.set_params(params)
    return model


def set_initial_params(model):
    n_classes = 2  
    n_features = 115  # Number of features in dataset
    model.classes_ = np.array([i for i in range(n_classes)])

    # Initialize the estimators (trees) with dummy values
    model.estimators_ = [
        RandomForestClassifier(n_estimators=1, max_depth=None).fit(np.zeros((1, n_features)), np.zeros(1)) for _ in
        range(model.n_estimators)]

    return model
