from typing import  Callable, Dict, List, Optional, Tuple, Union
from logging import WARNING
from flwr.common.logger import log
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
from dataclasses import dataclass
from flwr.server.client_manager import ClientManager
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy.aggregate import aggregate, weighted_loss_avg, aggregate_popper
from .strategy import Strategy
from collections import OrderedDict
from popper.util import Settings
from popper.tester import Tester
from popper.core import Clause
from popper.util import Settings, Stats
from logging import DEBUG
import logging
import numpy as np 
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FedPopper(Strategy):
    def __init__(
        self,
        settings: None,
        current_hypothesis = None,
        current_before = None,
        current_min_clause = None,
        stats = None,
        solver = None,
        grounder = None,
        constrainer = None,
        fraction_fit: float = 1.0,
        fraction_evaluate: float = 1.0,
        min_fit_clients: int = 2,
        min_evaluate_clients: int = 2,
        min_available_clients: int = 2,
        initial_parameters: Optional[Parameters] = None,
        fit_metrics_aggregation_fn: Optional[MetricsAggregationFn] = None,
    ) -> None:
        super().__init__()
        self.settings = settings
        self.fraction_fit = fraction_fit
        self.fraction_evaluate = fraction_evaluate
        self.min_fit_clients = min_fit_clients
        self.min_evaluate_clients = min_evaluate_clients
        self.min_available_clients = min_available_clients
        self.initial_parameters = initial_parameters
        self.fit_metrics_aggregation_fn = fit_metrics_aggregation_fn
        self.current_hypothesis = current_hypothesis
        self.current_before = current_before
        self.current_min_clause = current_min_clause
        self.solver = solver
        self.grounder = grounder
        self.constrainer = constrainer
        self.stats = stats if stats else Stats(log_best_programs = True)
    def __repr__(self) -> str:
        return "FedConstraints"
    
    def initialize_parameters(
        self, client_manager: ClientManager
    ) -> Optional[Parameters]:
        """Initialize global model parameters."""
        if self.current_hypothesis: 
            log(DEBUG,"Sending stored hypothesis to clients")
            return ndarrays_to_parameters(np.array(self.current_hypothesis, dtype="<U100"))
        log(WARNING,"No stored hypothesis, sending initial parameters.")

        #initial_parameters = self.initial_parameters
        #self.initial_parameters = None  # Don't keep initial parameters in memory
        return self.initial_parameters

    def configure_fit(
        self, server_round: int, parameters: Parameters, client_manager: ClientManager
    ) -> List[Tuple[ClientProxy, FitIns]]:
        """Configure the next round of training."""

        # Sample clients
        sample_size, min_num_clients = self.num_fit_clients(
            client_manager.num_available()
        )
        clients = client_manager.sample(
            num_clients=sample_size, min_num_clients=min_num_clients
        )

        # Create custom configs
        n_clients = len(clients)
        half_clients = n_clients // 2
        standard_config = {"lr": 0.001}
        higher_lr_config = {"lr": 0.003}
        fit_configurations = []
        for idx, client in enumerate(clients):
            if idx < half_clients:
                fit_configurations.append((client, FitIns(parameters, standard_config)))
            else:
                fit_configurations.append(
                    (client, FitIns(parameters, higher_lr_config))
                )
        return fit_configurations

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """Aggregate fit results using the ILP aggregation method."""
        
        # ✅ Step 1: Extract outcome pairs (E+, E-) and number of examples
        outcome_results = [
            (parameters_to_ndarrays(fit_res.parameters), fit_res.num_examples)
            for _, fit_res in results
        ]
        
        log(DEBUG, f"Received encoded outcomes from clients: {outcome_results}")
        # ✅ Step 2: Aggregate outcomes and generate new rules
        new_rules, solver, stats, min_clause, before = aggregate_popper(outcome_results, self.settings, self.solver, self.stats,self.current_hypothesis,self.current_min_clause, self.current_before)
        # ✅ Store the hypothesis for the next round
        if new_rules and len(new_rules[0]) > 0:
            self.current_hypothesis = new_rules
            self.current_before = before
            self.current_min_clause = min_clause
            #self.stats = updated_stats
            log(DEBUG,"✅ Updated current hypothesis and constraints for next round.")
        if solver and stats:
            self.stats = stats
            log(DEBUG,"✅ Updated current stats and solver for next round.")
    
        #log(DEBUG, f"Generated hypothesis (new rules) from constraints: {new_rules}")

        new_rules_ndarray = np.array(new_rules, dtype="<U100")

        # ✅ Step 3: Convert rules to Flower parameters
        parameters_aggregated = ndarrays_to_parameters(new_rules_ndarray)

        # ✅ Step 4: Aggregate custom metrics if provided
        metrics_aggregated = {}
        if self.fit_metrics_aggregation_fn:
            fit_metrics = [(res.num_examples, res.metrics) for _, res in results]
            metrics_aggregated = self.fit_metrics_aggregation_fn(fit_metrics)
        elif server_round == 1:
            log(WARNING, "No fit_metrics_aggregation_fn provided")

        return parameters_aggregated, metrics_aggregated
    

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

    def aggregate_evaluate(
    self,
    server_round: int,
    results: List[Tuple[ClientProxy, EvaluateRes]],
    failures: List[Union[Tuple[ClientProxy, EvaluateRes], BaseException]],
) -> Tuple[Optional[float], Dict[str, Scalar]]:
        """Aggregate evaluation losses using weighted average."""
        
        if not results:
            log(WARNING, "No evaluation results received! Returning default values.")
            return None, {}

        num_total_evaluation_examples = sum(evaluate_res.num_examples for _, evaluate_res in results)

        if num_total_evaluation_examples == 0:
            log(WARNING, "No valid examples for evaluation! Avoiding division by zero.")
            return None, {}

        loss_aggregated = weighted_loss_avg(
            [
                (evaluate_res.num_examples, evaluate_res.loss)
                for _, evaluate_res in results
            ]
        )
        metrics_aggregated = {}

        return loss_aggregated, metrics_aggregated


    def evaluate(
        self, server_round: int, parameters: Parameters
    ) -> Optional[Tuple[float, Dict[str, Scalar]]]:
        """Evaluate global model parameters using an evaluation function."""
        
        return None

    def num_fit_clients(self, num_available_clients: int) -> Tuple[int, int]:
        """Return sample size and required number of clients."""
        num_clients = int(num_available_clients * self.fraction_fit)
        return max(num_clients, self.min_fit_clients), self.min_available_clients

    def num_evaluation_clients(self, num_available_clients: int) -> Tuple[int, int]:
        """Use a fraction of available clients for evaluation."""
        num_clients = int(num_available_clients * self.fraction_evaluate)
        return max(num_clients, self.min_evaluate_clients), self.min_available_clients
