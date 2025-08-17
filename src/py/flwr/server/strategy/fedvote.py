# server_vote.py
from __future__ import annotations

import math
from logging import WARNING
from typing import Dict, List, Optional, Tuple, Union, Any

import numpy as np
import flwr as fl
from flwr.common import (
    EvaluateIns,
    FitIns,
    FitRes,
    MetricsAggregationFn,
    Parameters,
    Scalar,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)
from flwr.common.logger import log
from flwr.server.client_manager import ClientManager
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import Strategy
from flwr.server.strategy.aggregate import weighted_loss_avg


# -------------------------- helpers (robust to dtypes) --------------------------

def _to_float(x: Any) -> float:
    """Coerce a value to float; return NaN if not possible."""
    if isinstance(x, (int, float, np.integer, np.floating, bool)):
        return float(x)
    try:
        return float(str(x))
    except Exception:
        return math.nan


def _weighted_nanmean(pairs: List[Tuple[float, float]]) -> float:
    """Weighted mean ignoring NaNs. pairs: list of (weight, value)"""
    if not pairs:
        return math.nan
    num = 0.0
    den = 0.0
    for w, v in pairs:
        vv = _to_float(v)
        ww = float(w) if w is not None else 0.0
        if ww <= 0.0 or math.isnan(vv):
            continue
        num += ww * vv
        den += ww
    return (num / den) if den > 0 else math.nan


# ------------------------------- Strategy class --------------------------------

class FedVote(Strategy):
    """Federated majority-vote strategy over client prediction vectors.

    Protocol:
      • Clients send their GLOBAL prediction vector (0/1 for each example) in `fit()`.
      • Server does majority vote in `aggregate_fit` and stores the voted vector.
      • Server sends the voted vector as `parameters` in `configure_evaluate`.
      • Clients compute FL_acc_global locally in `evaluate()`.
    """

    def __init__(
        self,
        *,
        fraction_fit: float = 1.0,
        fraction_evaluate: float = 1.0,
        min_fit_clients: int = 2,
        min_evaluate_clients: int = 2,
        min_available_clients: int = 2,
        evaluate_fn=None,                  # optional server-side eval hook
        on_fit_config_fn=None,
        on_evaluate_config_fn=None,
        accept_failures: bool = True,
        initial_parameters: Optional[Parameters] = None,
        fit_metrics_aggregation_fn: Optional[MetricsAggregationFn] = None,
        evaluate_metrics_aggregation_fn: Optional[MetricsAggregationFn] = None,
    ) -> None:
        super().__init__()
        if (
            min_fit_clients > min_available_clients
            or min_evaluate_clients > min_available_clients
        ):
            log(
                WARNING,
                "min_available_clients must be >= min_fit_clients and min_evaluate_clients",
            )

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

    # ---------- required abstract methods ----------

    def initialize_parameters(self, client_manager: ClientManager) -> Optional[Parameters]:
        # Start with provided params or an empty vector; round 1 aggregate_fit will replace it
        if self.initial_parameters is not None:
            return self.initial_parameters
        return ndarrays_to_parameters([np.array([], dtype=np.int64)])

    def evaluate(
        self, server_round: int, parameters: Parameters
    ) -> Optional[Tuple[float, Dict[str, Scalar]]]:
        """Optional central eval; return None if unused."""
        if self.evaluate_fn is None:
            return None
        nds = parameters_to_ndarrays(parameters)
        out = self.evaluate_fn(server_round, nds, {})
        return None if out is None else (float(out[0]), out[1])

    def num_fit_clients(self, num_available_clients: int) -> Tuple[int, int]:
        n = max(int(num_available_clients * self.fraction_fit), self.min_fit_clients)
        return n, self.min_available_clients

    def num_evaluation_clients(self, num_available_clients: int) -> Tuple[int, int]:
        n = max(int(num_available_clients * self.fraction_evaluate), self.min_evaluate_clients)
        return n, self.min_available_clients

    def configure_fit(
        self, server_round: int, parameters: Parameters, client_manager: ClientManager
    ):
        cfg = self.on_fit_config_fn(server_round) if self.on_fit_config_fn else {}
        fit_ins = FitIns(parameters, cfg)
        sample_size, min_num = self.num_fit_clients(client_manager.num_available())
        clients = client_manager.sample(num_clients=sample_size, min_num_clients=min_num)
        return [(c, fit_ins) for c in clients]

    def configure_evaluate(
        self, server_round: int, parameters: Parameters, client_manager: ClientManager
    ):
        if self.fraction_evaluate == 0.0:
            return []
        cfg = self.on_evaluate_config_fn(server_round) if self.on_evaluate_config_fn else {}
        evaluate_ins = EvaluateIns(parameters, cfg)
        sample_size, min_num = self.num_evaluation_clients(client_manager.num_available())
        clients = client_manager.sample(num_clients=sample_size, min_num_clients=min_num)
        return [(c, evaluate_ins) for c in clients]

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        if not results:
            return None, {}
        if not self.accept_failures and failures:
            return None, {}

        # Collect prediction vectors from clients
        preds: List[np.ndarray] = []
        for _, fit_res in results:
            arrs = parameters_to_ndarrays(fit_res.parameters)
            if not arrs:
                continue
            # Expect one 1D array of 0/1 predictions
            p = np.asarray(arrs[0]).astype(np.int64, copy=False).ravel()
            preds.append(p)

        if not preds:
            return None, {}

        # Consistency check: all clients must predict for the same number of examples (same order!)
        L = len(preds[0])
        if any(len(p) != L for p in preds):
            raise ValueError("Prediction length mismatch across clients")

        # Majority vote
        P = np.stack(preds, axis=0)  # (n_clients, L)
        votes = P.sum(axis=0)
        maj = (P.shape[0] // 2) + 1
        agg = (votes >= maj).astype(np.int64)
        # 👇 add this log
        log(WARNING, f"[FedVote] Aggregated majority from {P.shape[0]} clients "
                     f"over {P.shape[1]} examples | mean_support={np.mean(votes)/max(P.shape[0],1):.3f}")
        # Metrics (diagnostics)
        metrics: Dict[str, Scalar] = {
            "mean_support": float(np.mean(votes / max(P.shape[0], 1))),
            "n_clients": float(P.shape[0]),
        }

        # Return voted vector as next-round parameters (clients will receive it in evaluate)
        return ndarrays_to_parameters([agg]), metrics

    def aggregate_evaluate(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, "EvaluateRes"]],
        failures: List[Union[Tuple[ClientProxy, "EvaluateRes"], BaseException]],
    ) -> Optional[Tuple[float, Dict[str, Scalar]]]:
        """Aggregate evaluation results (loss + metrics) robustly."""
        if not results:
            return None
        if not self.accept_failures and failures:
            return None

        # Weighted loss (by num_examples)
        loss_results: List[Tuple[int, float]] = []
        for _, ev in results:
            n = int(getattr(ev, "num_examples", 0) or 0)
            l = _to_float(getattr(ev, "loss", math.nan))
            if n > 0 and not math.isnan(l):
                loss_results.append((n, l))
        agg_loss = float(weighted_loss_avg(loss_results)) if loss_results else math.nan

        # Weighted, NaN-safe metrics aggregation
        all_keys = set()
        for _, ev in results:
            mdict = getattr(ev, "metrics", None)
            if isinstance(mdict, dict):
                all_keys.update(mdict.keys())

        metrics: Dict[str, Scalar] = {}
        for k in sorted(all_keys):
            pairs: List[Tuple[int, float]] = []
            for _, ev in results:
                n = int(getattr(ev, "num_examples", 0) or 0)
                mdict = getattr(ev, "metrics", None)
                if not isinstance(mdict, dict) or k not in mdict:
                    continue
                pairs.append((n, _to_float(mdict[k])))
            val = _weighted_nanmean(pairs)
            if not math.isnan(val):
                metrics[k] = float(val)

        return agg_loss, metrics


