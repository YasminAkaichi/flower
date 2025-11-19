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

from popper.constrain import Constrain
from popper.core import Clause, Literal, ConstVar
from popper.asp import ClingoGrounder, ClingoSolver
from dataclasses import dataclass
from flwr.server.client_manager import ClientManager
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy.aggregate import aggregate, weighted_loss_avg, aggregate_popper
from .strategy import Strategy
from collections import OrderedDict
from popper.util import Settings
from popper.tester import Tester

from popper.structural_tester import StructuralTester
from popper.core import Clause
from popper.util import Settings, Stats
from logging import DEBUG
import logging
import ast
import numpy as np 
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
from flwr.server.strategy.aggregate import aggregate_outcomes  # tu l’as déjà
# si besoin local:
DECODE = {1: "all", 2: "some", 3: "none"}

class FedPopper(Strategy):
    def __init__(
        self,
        settings: None,
        current_hypothesis = None,
        current_before = None,
        current_min_clause = 0,
        current_clause_size= 0, 
        stats = None,
        solver = None,
        grounder = None,
        constrainer = None,
        tester = None,
        seen_prog = None,
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
        self.current_clause_size = current_clause_size
        self.solver = solver if solver is not None else ClingoSolver(settings)
        self.grounder = grounder if grounder is not None else ClingoGrounder()
        self.constrainer = constrainer if constrainer is not None else Constrain()
        self.tester = tester if tester is not None else StructuralTester()
        self.stats : stats if stats is not None else Stats(log_best_programs=settings.info)
        self.seen_prog = seen_prog
        self.stats = stats 
        self.solution_params: Optional[Parameters] = None
        self.early_stop = False
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

        # === EARLY STOPPING ===
        if getattr(self, "early_stop", False):
            log(DEBUG, f"⏹️ Early stop triggered at round {server_round}. Ending FL loop.")
            return []      # ← FLORRRRR arrête ici automatiquement

        # === If not early-stopped → normal sampling ===
        sample_size, min_num_clients = self.num_fit_clients(
            client_manager.num_available()
        )
        clients = client_manager.sample(
            num_clients=sample_size,
            min_num_clients=min_num_clients,
        )

        n_clients = len(clients)
        half = n_clients // 2

        standard_cfg = {"lr": 0.001}
        high_cfg = {"lr": 0.003}

        fit_cfg = []
        for idx, client in enumerate(clients):
            cfg = standard_cfg if idx < half else high_cfg
            fit_cfg.append((client, FitIns(parameters, cfg)))

        return fit_cfg



    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """Aggregate fit results using the ILP aggregation method."""
        # 1) Collect outcomes from clients
        outcome_results = [
            (parameters_to_ndarrays(res.parameters), res.num_examples)
            for _, res in results
        ]

        # Decode outcomes
        encoded = [tuple(arr[0].tolist()) for arr, _ in outcome_results]
        decoded = [(DECODE[int(a)], DECODE[int(b)]) for (a,b) in encoded]

        ep_glob, en_glob = aggregate_outcomes(decoded)

        # 2) Step Popper : one iteration of while-loop
        new_rules, min_clause, before, clause_size, solver, solved, true_rules = aggregate_popper(
            outcome_results,
            self.settings,
            self.solver,
            self.grounder,
            self.constrainer,
            self.tester,
            self.stats,
            self.current_min_clause,
            self.current_before,
            self.current_hypothesis,
            self.current_clause_size
        )
        # 3) Update internal state for next round
        self.current_min_clause = min_clause
        self.current_before = before
        self.current_clause_size = clause_size
        self.solver = solver
        #self.early_stop = solved
        # Keep hypothesis ONLY if not empty
        if new_rules and len(new_rules[0])>0:
            #self.current_hypothesis = new_rules
            self.current_hypothesis = true_rules 

        # 4) Convert rules to Flower-format
        params = ndarrays_to_parameters(new_rules)

        # 5) Early stop if true solution
        if (ep_glob, en_glob) == ("all","none"):
            self.solution_params = params
            self.early_stop = True

        return params, {}
    

    def configure_evaluate(
    self, server_round: int, parameters: Parameters, client_manager: ClientManager
) -> List[Tuple[ClientProxy, EvaluateIns]]:

        """Configure the next round of evaluation."""

        # ✅ EARLY STOP: If the final program was found, we evaluate ONCE then stop.
        if getattr(self, "early_stop", False):
            log(DEBUG,f"⏹️ Early stop is active at round {server_round}: skipping evaluation.")
            return []

        if self.fraction_evaluate == 0.0:
            return []

        # ✅ Use final hypothesis if discovered, otherwise use current parameters
        params_for_eval = self.solution_params if self.solution_params is not None else parameters

        config = {}
        evaluate_ins = EvaluateIns(params_for_eval, config)

        # Sample clients
        sample_size, min_num_clients = self.num_evaluation_clients(client_manager.num_available())
        clients = client_manager.sample(
            num_clients=sample_size,
            min_num_clients=min_num_clients
        )

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