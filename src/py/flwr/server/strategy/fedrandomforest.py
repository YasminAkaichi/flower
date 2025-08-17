# Copyright 2020 Flower Labs GmbH. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Federated Averaging (FedAvg) [McMahan et al., 2016] strategy.

Paper: arxiv.org/abs/1602.05629
"""
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.ensemble import RandomForestClassifier
import pickle
import numpy as np
from flwr.common import Parameters, ndarrays_to_parameters
from logging import WARNING, DEBUG
from typing import Callable, Dict, List, Optional, Tuple, Union

from flwr.common import (
    EvaluateIns,
    EvaluateRes,
    FitIns,
    FitRes,
    MetricsAggregationFn,
    NDArrays,
    Parameters,
    Scalar,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)
from flwr.common.logger import log
from flwr.server.client_manager import ClientManager
from flwr.server.client_proxy import ClientProxy

from .aggregate import aggregate, aggregate_inplace, weighted_loss_avg, _parse_tree_from_string
from .strategy import Strategy
from sklearn.tree import export_text
import pickle
import numpy as np
WARNING_MIN_AVAILABLE_CLIENTS_TOO_LOW = """
Setting `min_available_clients` lower than `min_fit_clients` or
`min_evaluate_clients` can cause the server to fail when there are too few clients
connected to the server. `min_available_clients` must be set to a value larger
than or equal to the values of `min_fit_clients` and `min_evaluate_clients`.
"""


# pylint: disable=line-too-long
class FedRandomForest(Strategy):
    """Federated Averaging strategy.

    Implementation based on https://arxiv.org/abs/1602.05629

    Parameters
    ----------
    fraction_fit : float, optional
        Fraction of clients used during training. In case `min_fit_clients`
        is larger than `fraction_fit * available_clients`, `min_fit_clients`
        will still be sampled. Defaults to 1.0.
    fraction_evaluate : float, optional
        Fraction of clients used during validation. In case `min_evaluate_clients`
        is larger than `fraction_evaluate * available_clients`,
        `min_evaluate_clients` will still be sampled. Defaults to 1.0.
    min_fit_clients : int, optional
        Minimum number of clients used during training. Defaults to 2.
    min_evaluate_clients : int, optional
        Minimum number of clients used during validation. Defaults to 2.
    min_available_clients : int, optional
        Minimum number of total clients in the system. Defaults to 2.
    evaluate_fn : Optional[Callable[[int, NDArrays, Dict[str, Scalar]],Optional[Tuple[float, Dict[str, Scalar]]]]]
        Optional function used for validation. Defaults to None.
    on_fit_config_fn : Callable[[int], Dict[str, Scalar]], optional
        Function used to configure training. Defaults to None.
    on_evaluate_config_fn : Callable[[int], Dict[str, Scalar]], optional
        Function used to configure validation. Defaults to None.
    accept_failures : bool, optional
        Whether or not accept rounds containing failures. Defaults to True.
    initial_parameters : Parameters, optional
        Initial global model parameters.
    fit_metrics_aggregation_fn : Optional[MetricsAggregationFn]
        Metrics aggregation function, optional.
    evaluate_metrics_aggregation_fn : Optional[MetricsAggregationFn]
        Metrics aggregation function, optional.
    inplace : bool (default: True)
        Enable (True) or disable (False) in-place aggregation of model updates.
    """

    # pylint: disable=too-many-arguments,too-many-instance-attributes, line-too-long
    def __init__(
        self,
        *,
        
        fraction_fit: float = 1.0,
        fraction_evaluate: float = 1.0,
        min_fit_clients: int = 2,
        min_evaluate_clients: int = 2,
        min_available_clients: int = 2,
        tree_pool = [],
        evaluate_fn: Optional[
            Callable[
                [int, NDArrays, Dict[str, Scalar]],
                Optional[Tuple[float, Dict[str, Scalar]]],
            ]
        ] = None,
        on_fit_config_fn: Optional[Callable[[int], Dict[str, Scalar]]] = None,
        on_evaluate_config_fn: Optional[Callable[[int], Dict[str, Scalar]]] = None,
        accept_failures: bool = True,
        initial_parameters: Optional[Parameters] = None,
        fit_metrics_aggregation_fn: Optional[MetricsAggregationFn] = None,
        evaluate_metrics_aggregation_fn: Optional[MetricsAggregationFn] = None,
        inplace: bool = True,
    ) -> None:
        super().__init__()

        if (
            min_fit_clients > min_available_clients
            or min_evaluate_clients > min_available_clients
        ):
            log(WARNING, WARNING_MIN_AVAILABLE_CLIENTS_TOO_LOW)
        self.tree_pool = tree_pool
        self.fraction_fit = fraction_fit
        self.fraction_evaluate = fraction_evaluate
        self.min_fit_clients = min_fit_clients
        self.min_evaluate_clients = min_evaluate_clients
        self.min_available_clients = min_available_clients
        self.evaluate_fn = evaluate_fn
        self.on_fit_config_fn = on_fit_config_fn
        self.on_evaluate_config_fn = on_evaluate_config_fn
        self.accept_failures = accept_failures
        self.initial_parameters = initial_parameters
        self.fit_metrics_aggregation_fn = fit_metrics_aggregation_fn
        self.evaluate_metrics_aggregation_fn = evaluate_metrics_aggregation_fn
        self.inplace = inplace

    def __repr__(self) -> str:
        """Compute a string representation of the strategy."""
        rep = f"FedAvg(accept_failures={self.accept_failures})"
        return rep

    def num_fit_clients(self, num_available_clients: int) -> Tuple[int, int]:
        """Return the sample size and the required number of available clients."""
        num_clients = int(num_available_clients * self.fraction_fit)
        return max(num_clients, self.min_fit_clients), self.min_available_clients

    def num_evaluation_clients(self, num_available_clients: int) -> Tuple[int, int]:
        """Use a fraction of available clients for evaluation."""
        num_clients = int(num_available_clients * self.fraction_evaluate)
        return max(num_clients, self.min_evaluate_clients), self.min_available_clients

    def initialize_parameters(
        self, client_manager: ClientManager
    ) -> Optional[Parameters]:
        """Initialize global model parameters."""
        initial_parameters = self.initial_parameters
        self.initial_parameters = None  # Don't keep initial parameters in memory
        return initial_parameters

    def evaluate(
        self, server_round: int, parameters: Parameters
    ) -> Optional[Tuple[float, Dict[str, Scalar]]]:
        """Evaluate model parameters using an evaluation function."""
        if self.evaluate_fn is None:
            # No evaluation function provided
            return None
        parameters_ndarrays = parameters_to_ndarrays(parameters)
        eval_res = self.evaluate_fn(server_round, parameters_ndarrays, {})
        if eval_res is None:
            return None
        loss, metrics = eval_res
        return loss, metrics

    def configure_fit(
        self, server_round: int, parameters: Parameters, client_manager: ClientManager
    ) -> List[Tuple[ClientProxy, FitIns]]:
        """Configure the next round of training."""
        config = {}
        if self.on_fit_config_fn is not None:
            # Custom fit config function provided
            config = self.on_fit_config_fn(server_round)
        fit_ins = FitIns(parameters, config)

        # Sample clients
        sample_size, min_num_clients = self.num_fit_clients(
            client_manager.num_available()
        )
        clients = client_manager.sample(
            num_clients=sample_size, min_num_clients=min_num_clients
        )

        # Return client/config pairs
        return [(client, fit_ins) for client in clients]

    def configure_evaluate(
        self, server_round: int, parameters: Parameters, client_manager: ClientManager
    ) -> List[Tuple[ClientProxy, EvaluateIns]]:
        """Configure the next round of evaluation."""
        # Do not configure federated evaluation if fraction eval is 0.
        if self.fraction_evaluate == 0.0:
            return []

        # Parameters and config
        config = {}
        if self.on_evaluate_config_fn is not None:
            # Custom evaluation config function provided
            config = self.on_evaluate_config_fn(server_round)
        evaluate_ins = EvaluateIns(parameters, config)

        # Sample clients
        sample_size, min_num_clients = self.num_evaluation_clients(
            client_manager.num_available()
        )
        clients = client_manager.sample(
            num_clients=sample_size, min_num_clients=min_num_clients
        )

        # Return client/config pairs
        return [(client, evaluate_ins) for client in clients]




    def aggregate_fit(self, server_round, results, failures):
        trees = []
        print(f"\n🔄 Aggregating round {server_round}")

        for idx, (_, fit_res) in enumerate(results):
            try:
                tree_array = parameters_to_ndarrays(fit_res.parameters)[0]
                tree_bytes = tree_array.tobytes()
                payload = pickle.loads(tree_bytes)

                # ✅ Extraire uniquement le modèle
                model = payload["model"]
                acc = fit_res.metrics.get("local_train_acc", 0.0)
                print(f"📊 Client {idx} accuracy: {acc:.4f}")

                if isinstance(model, DecisionTreeClassifier):
                    trees.append(model)
                    print(f"✅ Arbre du client {idx} conservé (acc={acc:.2f})")

                    # 🔍 Affichage de l’arbre local
                    feature_names = [f"x{i}" for i in range(model.n_features_in_)]
                    print(f"\n🌲 Arbre reçu du client {idx} :")
                    print(export_text(model, feature_names=feature_names))

                elif isinstance(model, RandomForestClassifier):
                    print(f"📦 Client {idx} a envoyé une forêt. Extraction des arbres internes...")
                    for j, tree in enumerate(model.estimators_):
                        trees.append(tree)
                        feature_names = [f"x{i}" for i in range(tree.n_features_in_)]
                        print(f"\n🌳 Arbre {j+1} du client {idx} :")
                        print(export_text(tree, feature_names=feature_names))

            except Exception as e:
                print(f"⚠️ Erreur lors du traitement de l'arbre du client {idx}: {e}")

        print(f"\n✅ Total des arbres conservés : {len(trees)}")

        if not trees:
            print("❌ Aucun arbre valide reçu. On retourne un modèle vide.")
            return ndarrays_to_parameters([]), {}

        # ✅ Création du modèle agrégé
        rf = RandomForestClassifier(n_estimators=len(trees))
        rf.estimators_ = trees
        rf.classes_ = np.unique(np.concatenate([tree.classes_ for tree in trees]))
        rf.n_classes_ = len(rf.classes_)

        print("\n🌲 Random Forest agrégée finale :")
        for i, tree in enumerate(rf.estimators_):
            feature_names = [f"x{j}" for j in range(tree.n_features_in_)]
            print(f"\n🧩 Arbre {i+1} :\n{export_text(tree, feature_names=feature_names)}")

        rf_bytes = pickle.dumps(rf)
        rf_array = np.frombuffer(rf_bytes, dtype=np.uint8)
        return ndarrays_to_parameters([rf_array]), {}





    """

    def fit(self, parameters, config):
        self.model.fit(self.X_train, self.y_train)
        # 🎯 Évaluer la performance locale juste après le training
        y_pred_train = self.model.predict(self.X_train)
        local_train_acc = accuracy_score(self.y_train, y_pred_train)
        print(f"\n📈 [Client] Accuracy locale sur le jeu d'entraînement : {local_train_acc:.4f}")
    
        feature_names = [f"x{i}" for i in range(self.model.n_features_in_)]
        tree_str = export_text(self.model, feature_names=feature_names)
        print("\n[INFO] Arbre local appris par le client :")
        print(tree_str)
        return [np.array([tree_str], dtype="<U1000")], len(self.X_train), {}
    def aggregate_fit(self, server_round, results, failures):
        trees = []
        print(f"\n🔄 Aggregating round {server_round}")

        for idx, (_, fit_res) in enumerate(results):
            try:
                tree_array = parameters_to_ndarrays(fit_res.parameters)[0]
                tree_list = tree_array.tolist()

                print(f"\n🌲 Tree(s) from Client {idx} (Round {server_round}):")
                for i, tree_str in enumerate(tree_list):
                    print(f"📜 Client {idx} Tree {i}:\n{tree_str}")
                    parsed_tree = _parse_tree_from_string(tree_str)
                    trees.append(parsed_tree)

            except Exception as e:
                print(f"⚠️ Failed to process tree from client {idx}: {e}")

        print(f"\n✅ Total trees parsed into models: {len(trees)}")

        if not trees:
            print("❌ No valid trees received. Returning empty model.")
            return ndarrays_to_parameters([]), {}

        # Aggregate into a RandomForest
        rf = RandomForestClassifier(n_estimators=len(trees))
        rf.estimators_ = trees
        rf.classes_ = np.unique(np.concatenate([tree.classes_ for tree in trees]))
        rf.n_classes_ = len(rf.classes_)

        print("\n🌲 Aggregated Random Forest:")
        for i, tree in enumerate(rf.estimators_):
            feature_names = [f"x{j}" for j in range(tree.n_features_in_)]
            print(f"\n🧩 Tree {i}:\n{export_text(tree, feature_names=feature_names)}")

        # Serialize the forest
        rf_bytes = pickle.dumps(rf)
        rf_array = np.frombuffer(rf_bytes, dtype=np.uint8)
        return ndarrays_to_parameters([rf_array]), {}

   """
    

    def aggregate_evaluate(
    self,
    server_round: int,
    results: List[Tuple[ClientProxy, EvaluateRes]],
    failures: List[Union[Tuple[ClientProxy, EvaluateRes], BaseException]],
) -> Tuple[Optional[float], Dict[str, Scalar]]:
        """Aggregate evaluation metrics from clients using average best_accuracy."""
        if not results:
            return None, {}

        total_examples = sum(res.num_examples for _, res in results)
        avg_accuracy = sum(
            res.num_examples * res.metrics.get("best_accuracy", 0.0)
            for _, res in results
        ) / total_examples

        return 0.0, {"accuracy": avg_accuracy}


    
    def configure_evaluate(
        self, server_round: int, parameters: Parameters, client_manager: ClientManager
    ) -> List[Tuple[ClientProxy, EvaluateIns]]:
        """Configure the next round of evaluation."""
        if self.fraction_evaluate == 0.0:
            return []
        config = {}
        evaluate_ins = EvaluateIns(parameters, config)

        # Sample clients
        sample_size, min_num_clients = self.num_evaluation_clients(
            client_manager.num_available()
        )
        clients = client_manager.sample(
            num_clients=sample_size, min_num_clients=min_num_clients
        )

        # Return client/config pairs
        return [(client, evaluate_ins) for client in clients]
