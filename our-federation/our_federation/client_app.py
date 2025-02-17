"""our-federation: A Flower / sklearn app."""

import warnings

from sklearn.metrics import log_loss
import numpy as np
from flwr.client import ClientApp, NumPyClient
from flwr.common import Context

from our_federation.task import (
    get_model,
    get_model_params,
    load_data,
    set_initial_params,
    set_model_params,
)

class FlowerClient(NumPyClient):
    def __init__(self, model, X_train, X_test, y_train, y_test):
        self.model = model
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test

    def fit(self, parameters, config):
        set_model_params(self.model, parameters)
        self.model.fit(self.X_train, self.y_train)
        return get_model_params(self.model), len(self.X_train), {}

    def evaluate(self, parameters, config):
        set_model_params(self.model, parameters)

        loss = log_loss(self.y_test, self.model.predict_proba(self.X_test),  labels=np.unique(self.y_test))
        accuracy = self.model.score(self.X_test, self.y_test)

        return loss, len(self.X_test), {"accuracy": accuracy}


def client_fn(context: Context):
    partition_id = context.node_config["partition-id"]
    num_partitions = context.node_config["num-partitions"]

    X_train, X_test, y_train, y_test = load_data(partition_id, num_partitions, 0.1)

    # Create RandomForestClassifier Model
    n_estimators = context.run_config["n_estimators"]
    max_depth = context.run_config.get("max_depth", None)
    model = get_model(n_estimators, max_depth)

    # Setting initial parameters, akin to model.compile for keras models
    set_initial_params(model)
    print(f'Client {partition_id} is setup.')
    return FlowerClient(model, X_train, X_test, y_train, y_test).to_client()


# Flower ClientApp
app = ClientApp(client_fn=client_fn)
