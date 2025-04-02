from logging import WARNING
from flwr.common.logger import log
from functools import reduce
from typing import Any, Callable, List, Tuple
import torch 
import numpy as np
from flwr.common import FitRes, NDArray, NDArrays, parameters_to_ndarrays, ndarray_to_bytes
from flwr.server.client_proxy import ClientProxy
from flwr.common.logger import log
from logging import INFO, WARN
from typing import List, Tuple
from functools import reduce
from flwr.common import NDArrays
from typing import List, Tuple
from popper.asp import ClingoGrounder, ClingoSolver
from popper.loop import build_rules, ground_rules
from popper.constrain import Constrain
from popper.generate import generate_program
from logging import DEBUG 
from collections import Counter
from popper.util import Stats
import logging
from popper.tester import Tester
from popper.constrain import Constrain
from popper.core import Clause, Literal, ConstVar
from popper.asp import ClingoGrounder, ClingoSolver
from popper.util import load_kbpath, parse_settings,  Settings, Stats
from popper.loop import Outcome, build_rules, decide_outcome, ground_rules, Con,calc_score
from clingo import Function, Number, String
from typing import List, Tuple, Dict
import re 
# 🔹 Logging Setup
logging.basicConfig(level=logging.DEBUG)
log = logging.getLogger(__name__)

OUTCOME_ENCODING = {"ALL": 1, "SOME": 2, "NONE": 3}
OUTCOME_DECODING = {1: "ALL", 2: "SOME", 3: "NONE"}




OUTCOME_TO_CONSTRAINTS = {
        (Outcome.ALL, Outcome.NONE)  : (Con.BANISH,),
        (Outcome.ALL, Outcome.SOME)  : (Con.GENERALISATION,),
        (Outcome.SOME, Outcome.NONE) : (Con.SPECIALISATION,),
        (Outcome.SOME, Outcome.SOME) : (Con.SPECIALISATION, Con.GENERALISATION),
        (Outcome.NONE, Outcome.NONE) : (Con.SPECIALISATION, Con.REDUNDANCY),
        (Outcome.NONE, Outcome.SOME) : (Con.SPECIALISATION, Con.REDUNDANCY, Con.GENERALISATION)
    }



# 🔹 Example Aggregation Table (Modify as Needed)
AGGREGATION_TABLE_pos_outcome = {
    ("all", "all"): "all",
    ("all", "some"): "all",
    ("all", "none"): "all",
    ("some", "some"): "some",
    ("some", "none"): "some",
    ("none", "none"): "none",
}

AGGREGATION_TABLE_neg_outcome = {
    ("some", "some"): "some",
    ("some", "none"): "some",
    ("none", "some"): "none",
    ("none", "none"): "none",
}



def aggregate_outcomes(outcomes: List[Tuple[str, str]]) -> Tuple[str, str]:
    """
    Agrège une liste de paires d'outcomes (E+, E-) en utilisant des règles spécifiques pour chaque cas.
    
    :param outcomes: Liste de tuples contenant les outcomes sous forme de chaînes ('all', 'some', 'none').
    :return: Une paire unique (E+, E-) agrégée.
    """
    valid_outcomes = [o for o in outcomes if len(o) == 2]
    if not valid_outcomes:
        log.warning("⚠️ No valid outcomes received! Using default (NONE, NONE).")
        return ("none", "none")  # Default outcome

    aggregated_E_plus = valid_outcomes[0][0]
    aggregated_E_minus = valid_outcomes[0][1]

    #aggregated_E_plus = outcomes[0][0]  # Prendre le premier E+
    #aggregated_E_minus = outcomes[0][1]  # Prendre le premier E-

    for E_plus, E_minus in outcomes[1:]:
        # Agrégation spécifique pour E+
        aggregated_E_plus = AGGREGATION_TABLE_pos_outcome.get(
            (aggregated_E_plus, E_plus), aggregated_E_plus
        )
        # Agrégation spécifique pour E-
        aggregated_E_minus = AGGREGATION_TABLE_neg_outcome.get(
            (aggregated_E_minus, E_minus), aggregated_E_minus
        )

    return (aggregated_E_plus, aggregated_E_minus)


def transform_rule_to_tester_format(rule_str):
    log.debug(f"🔍 Transforming rule: {rule_str}")

    try:
        # ✅ Split head and body correctly
        head_body = rule_str.split(":-")
        if len(head_body) != 2:
            raise ValueError(f"Invalid rule format: {rule_str}")

        head_str = head_body[0].strip()
        body_str = head_body[1].strip()

        # ✅ **Fix: Properly extract body literals using regex**
        body_literals = re.findall(r'\w+\(.*?\)', body_str)

        log.debug(f"🔹 Parsed head: {head_str}")
        log.debug(f"🔹 Parsed body literals: {body_literals}")

        # ✅ Convert to Literal objects (assuming `Literal.from_string` exists)
        head = Literal.from_string(head_str)
        body = tuple(Literal.from_string(lit) for lit in body_literals)

        formatted_rule = (head, body)
        log.debug(f"✅ Formatted rule: {formatted_rule}")

        return formatted_rule
    except Exception as e:
        log.error(f"❌ Error transforming rule: {rule_str} → {e}")
        return None  # Return None to indicate failure
    

def aggregate_popper0(outcome_list, settings, solver, grounder, constrainer, tester, stats,
                     current_hypothesis, current_min_clause, current_before, clause_size):
    """
    Tries to generate a valid hypothesis by increasing clause size if needed.
    """
    

    log.info(f"Received Outcomes: {outcome_list}")

    outcomes = [outcomes for outcomes, _ in outcome_list]
    aggregated_outcome = aggregate_outcomes([tuple(outcome[0]) for outcome in outcomes])

    decoded_outcome = (
        OUTCOME_DECODING[int(aggregated_outcome[0])],
        OUTCOME_DECODING[int(aggregated_outcome[1])],
    )

    normalized_outcome = (
        Outcome.ALL if decoded_outcome[0].upper() == "ALL" else Outcome.SOME if decoded_outcome[0].upper() == "SOME" else Outcome.NONE,
        Outcome.ALL if decoded_outcome[1].upper() == "ALL" else Outcome.SOME if decoded_outcome[1].upper() == "SOME" else Outcome.NONE,
    )

    log.info(f"\u2705 Normalized Outcome: {normalized_outcome}")

    stop = False
    current_rules = []
    if current_hypothesis is None:
        log.debug("Generating new initial hypothesis")
        with stats.duration('generate'):
            model = solver.get_model()
            if not model:
                log.warning("⚠️ No model generated — increasing clause_size")
                clause_size += 1
            current_rules, current_before, current_min_clause = generate_program(model)
            current_hypothesis = current_rules
            stats.register_hypothesis(current_rules)
    else:
        try:
            hypo = np.array(current_hypothesis, dtype="<U100")
            received_rules = hypo[0].tolist()
            parsed_rules = [transform_rule_to_tester_format(rule) for rule in received_rules]
            current_rules = [rule for rule in parsed_rules if rule is not None]
            stats.register_hypothesis(current_rules)
        except Exception as e:
            log.error(f"Error processing hypothesis: {e}")
            current_rules = []
    while clause_size <= settings.max_literals:
        log.debug(f"******************** MAX LITERALS: {clause_size} ********************")

        solver.update_number_of_literals(clause_size)
        stats.update_num_literals(clause_size)
        

        with stats.duration("generate"):
            model = solver.get_model()
            if not model:
                log.warning("No model after constraints. Breaking loop.")
                rule_ndarray = np.array([], dtype="<U100")
                clause_size += 1
                return [rule_ndarray], stats, current_min_clause, current_before, clause_size
            current_rules, before, min_clause = generate_program(model)
            current_before = before
            current_min_clause = min_clause
            stats.register_hypothesis(current_rules)
        if normalized_outcome == (Outcome.ALL, Outcome.NONE):
            log.debug("Perfect hypothesis found")
            stop = True
            stats.register_best_hypothesis(current_rules)
            new_rules_bytes = [Clause.to_code(rule) for rule in current_rules]
            new_rules_ndarray = np.array(new_rules_bytes, dtype="<U100")
            return [new_rules_ndarray], stats, current_min_clause, current_before, clause_size

        with stats.duration('build'):
            constraints = build_rules(settings, stats, constrainer, tester,
                                      current_rules, current_before, current_min_clause, normalized_outcome)

        with stats.duration('ground'):
            grounded = ground_rules(stats, grounder, solver.max_clauses, solver.max_vars, constraints)

        with stats.duration('add'):
            solver.add_ground_clauses(grounded)


        new_rules_bytes = [Clause.to_code(rule) for rule in current_rules]
        new_rules_ndarray = np.array(new_rules_bytes, dtype="<U100")
        return [new_rules_ndarray], stats, current_min_clause, current_before, clause_size

    log.warning("❌ Max clause size reached. No valid model found.")
    return [np.array([], dtype="<U100")], stats, current_min_clause, current_before, clause_size


def aggregate_popper1(
    outcome_list: List[Tuple[int, int]],
    settings,
    solver,
    grounder,
    constrainer,
    tester,
    stats,
    current_hypothesis,
    current_min_clause,
    current_before,
    clause_size,
):
    """
    Aggregate constraints based on outcome pairs (E+, E-), generate new constraints,
    and then use them to generate a new hypothesis (rules),
    mimicking Popper's inner while loop.
    """
    log.info(f"Received Outcomes: {outcome_list}")

    # Step 1: Aggregate and normalize outcomes
    outcomes = [outcome for outcome, _ in outcome_list]
    aggregated_outcome = aggregate_outcomes([tuple(outcome[0]) for outcome in outcomes])
    decoded_outcome = (
        OUTCOME_DECODING[int(aggregated_outcome[0])],
        OUTCOME_DECODING[int(aggregated_outcome[1])],
    )
    normalized_outcome = (
        Outcome.ALL if decoded_outcome[0].upper() == "ALL" else Outcome.SOME if decoded_outcome[0].upper() == "SOME" else Outcome.NONE,
        Outcome.ALL if decoded_outcome[1].upper() == "ALL" else Outcome.SOME if decoded_outcome[1].upper() == "SOME" else Outcome.NONE,
    )

    log.info(f"\u2705 Normalized Outcome: {normalized_outcome}")

    # Step 2: Set clause size
    #solver.update_number_of_literals(clause_size)
    #stats.update_num_literals(clause_size)
    current_rules = None


    solver.update_number_of_literals(clause_size)
    stats.update_num_literals(clause_size)
 
    with stats.duration("generate"):
        model = solver.get_model()
        if not model:
            log.warning("No model after constraints. Will increase clause size next round.")
            return [np.array([], dtype="<U100")], stats, current_min_clause, current_before, clause_size + 1

        current_rules, before, min_clause = generate_program(model)
        current_before = before
        current_min_clause = min_clause
        stats.register_hypothesis(current_rules)

    if normalized_outcome == (Outcome.ALL, Outcome.NONE):
        stats.register_best_hypothesis(current_rules)
        stats.register_completion()
        rule_bytes = [Clause.to_code(rule) for rule in current_rules]
        return [np.array(rule_bytes, dtype="<U100")], stats, current_min_clause, current_before, clause_size

    with stats.duration("build"):
        constraints = build_rules(settings, stats, constrainer, tester, current_rules, current_before, current_min_clause, normalized_outcome)
        log.debug(f"📭 Constraints generated: {constraints}")
    
    if not constraints:
        log.warning("No constraints generated. Will retry next round.")
        return [np.array(rule_bytes, dtype="<U100")], stats, current_min_clause, current_before, clause_size + 1

    with stats.duration("ground"):
        grounded_constraints = ground_rules(stats, grounder, solver.max_clauses, solver.max_vars, constraints)

    with stats.duration("add"):
        solver.add_ground_clauses(grounded_constraints)

    rule_bytes = [Clause.to_code(rule) for rule in current_rules]
    return [np.array(rule_bytes, dtype="<U100")], stats, current_min_clause, current_before, clause_size

def aggregate_popper(outcome_list: List[Tuple[int, int]], settings, solver, grounder, constrainer,tester, stats, current_hypothesis, current_min_clause, current_before, clause_size):
    """
    Aggregate constraints based on outcome pairs (E+, E-), generate new constraints,
    and then use them to generate a new hypothesis (rules).
    """
    
    log.info(f"Received Outcomes: {outcome_list}")
    
    # ✅ Step 1: Aggregate Outcomes
    outcomes = [outcomes for outcomes, _ in outcome_list]
    aggregated_outcome = aggregate_outcomes([tuple(outcome[0]) for outcome in outcomes])
    log.info(f"✅ Final Aggregated Outcome: {aggregated_outcome}")
    
    # ✅ Step 2: Normalize Outcome
    decoded_outcome = (
        OUTCOME_DECODING[int(aggregated_outcome[0])],
        OUTCOME_DECODING[int(aggregated_outcome[1])]
    )
    
    normalized_outcome = (
        Outcome.ALL if decoded_outcome[0].upper() == "ALL" else Outcome.SOME if decoded_outcome[0].upper() == "SOME" else Outcome.NONE,
        Outcome.ALL if decoded_outcome[1].upper() == "ALL" else Outcome.SOME if decoded_outcome[1].upper() == "SOME" else Outcome.NONE
    )
    
    log.info(f"✅ Final Aggregated Outcome: {normalized_outcome}")

    
    if current_hypothesis is None: 
        log.debug("Generate new first hypothesis")
        solver.update_number_of_literals(clause_size)
        stats.update_num_literals(clause_size)
    # Generate Hypothesis 
        with stats.duration('generate'):
            model = solver.get_model()
            if not model:
                log.debug("No model in solver")
                clause_size += 1
                empty_rule_array = np.array([], dtype="<U100")
                return [empty_rule_array], stats, current_min_clause, current_before, clause_size
            
            (current_rules, before, min_clause) = generate_program(model)
            current_hypothesis = current_rules
            current_before = before
            current_min_clause = min_clause
            stats.register_hypothesis(current_rules)
    else:
        log.debug("Use the existing hypothesis")
        try:
            # ✅ Convert received rules into Clause objects
            hypo = np.array(current_hypothesis, dtype="<U100") 
            received_rules = hypo[0].tolist()
            parsed_rules = [transform_rule_to_tester_format(rule) for rule in received_rules]
            #
            # 🔹 Remove any None values (failed transformations)
            current_rules = [rule for rule in parsed_rules if rule is not None]
            stats.register_hypothesis(current_rules)
            log.debug(f"✅ Updated client hypothesis: {current_rules}")
        except Exception as e:
            log.error(f"❌ Error processing received rules: {e}")
            current_rules = []  # Reset to empty if parsing fails

    log.debug(f"🔍 Rules before applying constraints: {current_rules}")     
     
    
    log.info(f"🔹 Normalized Outcome: {normalized_outcome}")

    if normalized_outcome == (Outcome.ALL, Outcome.NONE):
        print(" /!\ /!\ /!\ /!\ /!\ /!\ /!\ /!\ /!\ /!\ /!\ /!\ /!\ ")
        print(" ")
        print(normalized_outcome)
        print(" ")
        print(" /!\ /!\ /!\ /!\ /!\ /!\ /!\ /!\ /!\ /!\ /!\ /!\ /!\ ")
        log.debug("get_out_rounds, we okay")
        stats.register_best_hypothesis(current_rules)
        new_rules_bytes = [Clause.to_code(rule) for rule in current_rules]
        new_rules_ndarray = np.array(new_rules_bytes, dtype="<U100")
        return [new_rules_ndarray], stats, current_min_clause, current_before, clause_size
    

    # ✅ Step 7: Generate Constraints with Stats Tracking
    with stats.duration('build'):
        constraints = build_rules(settings, stats, constrainer, tester, current_rules, current_before, current_min_clause, normalized_outcome)

    log.debug(f"🔍 Generated Constraints: {constraints}")

    with stats.duration('ground'):
        grounded_constraints = ground_rules(stats, grounder, solver.max_clauses, solver.max_vars, constraints)
    
    log.debug(f"🔍 Generated Constraints: {solver.get_model()}")
    with stats.duration('add'):
        solver.add_ground_clauses(grounded_constraints)
    

    with stats.duration('generate'):
        model = solver.get_model()
        if not model: 
            log.warning("No model generated after constraints in solver")
            rule_bytes = [Clause.to_code(rule) for rule in current_rules]
            rule_ndarray = np.array(rule_bytes, dtype="<U100")
            clause_size += 1
            return [rule_ndarray], stats, current_min_clause, current_before, clause_size

        current_rules, before, min_clause = generate_program(solver.get_model())
        current_before = before
        current_min_clause = min_clause

    new_rules_bytes = [Clause.to_code(rule) for rule in current_rules]

    new_rules_ndarray = np.array(new_rules_bytes, dtype="<U100")

    log.info(f"🚀 Sending {len(new_rules_ndarray)} rules to clients.")

    return [new_rules_ndarray], stats, current_min_clause, current_before, clause_size


def aggregate(results: List[Tuple[NDArrays, int]]) -> NDArrays:
    """Compute weighted average."""
    # Calculate the total number of examples used during training
    num_examples_total = sum(num_examples for (_, num_examples) in results)

    # Create a list of weights, each multiplied by the related number of examples
    weighted_weights = [
        [layer * num_examples for layer in weights] for weights, num_examples in results
    ]

    # Compute average weights of each layer
    weights_prime: NDArrays = [
        reduce(np.add, layer_updates) / num_examples_total
        for layer_updates in zip(*weighted_weights)
    ]
    return weights_prime






def aggregate_inplace(results: List[Tuple[ClientProxy, FitRes]]) -> NDArrays:
    """Compute in-place weighted average."""
    # Count total examples
    num_examples_total = sum(fit_res.num_examples for (_, fit_res) in results)

    # Compute scaling factors for each result
    scaling_factors = [
        fit_res.num_examples / num_examples_total for _, fit_res in results
    ]

    # Let's do in-place aggregation
    # Get first result, then add up each other
    params = [
        scaling_factors[0] * x for x in parameters_to_ndarrays(results[0][1].parameters)
    ]
    for i, (_, fit_res) in enumerate(results[1:]):
        res = (
            scaling_factors[i + 1] * x
            for x in parameters_to_ndarrays(fit_res.parameters)
        )
        params = [reduce(np.add, layer_updates) for layer_updates in zip(params, res)]

    return params


def aggregate_median(results: List[Tuple[NDArrays, int]]) -> NDArrays:
    """Compute median."""
    # Create a list of weights and ignore the number of examples
    weights = [weights for weights, _ in results]

    # Compute median weight of each layer
    median_w: NDArrays = [
        np.median(np.asarray(layer), axis=0) for layer in zip(*weights)
    ]
    return median_w


def aggregate_krum(
    results: List[Tuple[NDArrays, int]], num_malicious: int, to_keep: int
) -> NDArrays:
    """Choose one parameter vector according to the Krum function.

    If to_keep is not None, then MultiKrum is applied.
    """
    # Create a list of weights and ignore the number of examples
    weights = [weights for weights, _ in results]

    # Compute distances between vectors
    distance_matrix = _compute_distances(weights)

    # For each client, take the n-f-2 closest parameters vectors
    num_closest = max(1, len(weights) - num_malicious - 2)
    closest_indices = []
    for distance in distance_matrix:
        closest_indices.append(
            np.argsort(distance)[1 : num_closest + 1].tolist()  # noqa: E203
        )

    # Compute the score for each client, that is the sum of the distances
    # of the n-f-2 closest parameters vectors
    scores = [
        np.sum(distance_matrix[i, closest_indices[i]])
        for i in range(len(distance_matrix))
    ]

    if to_keep > 0:
        # Choose to_keep clients and return their average (MultiKrum)
        best_indices = np.argsort(scores)[::-1][len(scores) - to_keep :]  # noqa: E203
        best_results = [results[i] for i in best_indices]
        return aggregate(best_results)

    # Return the model parameters that minimize the score (Krum)
    return weights[np.argmin(scores)]


# pylint: disable=too-many-locals
def aggregate_bulyan(
    results: List[Tuple[NDArrays, int]],
    num_malicious: int,
    aggregation_rule: Callable,  # type: ignore
    **aggregation_rule_kwargs: Any,
) -> NDArrays:
    """Perform Bulyan aggregation.

    Parameters
    ----------
    results: List[Tuple[NDArrays, int]]
        Weights and number of samples for each of the client.
    num_malicious: int
        The maximum number of malicious clients.
    aggregation_rule: Callable
        Byzantine resilient aggregation rule used as the first step of the Bulyan
    aggregation_rule_kwargs: Any
        The arguments to the aggregation rule.

    Returns
    -------
    aggregated_parameters: NDArrays
        Aggregated parameters according to the Bulyan strategy.
    """
    byzantine_resilient_single_ret_model_aggregation = [aggregate_krum]
    # also GeoMed (but not implemented yet)
    byzantine_resilient_many_return_models_aggregation = []  # type: ignore
    # Brute, Medoid (but not implemented yet)

    num_clients = len(results)
    if num_clients < 4 * num_malicious + 3:
        raise ValueError(
            "The Bulyan aggregation requires then number of clients to be greater or "
            "equal to the 4 * num_malicious + 3. This is the assumption of this method."
            "It is needed to ensure that the method reduces the attacker's leeway to "
            "the one proved in the paper."
        )
    selected_models_set: List[Tuple[NDArrays, int]] = []

    theta = len(results) - 2 * num_malicious
    beta = theta - 2 * num_malicious

    for _ in range(theta):
        best_model = aggregation_rule(
            results=results, num_malicious=num_malicious, **aggregation_rule_kwargs
        )
        list_of_weights = [weights for weights, num_samples in results]
        # This group gives exact result
        if aggregation_rule in byzantine_resilient_single_ret_model_aggregation:
            best_idx = _find_reference_weights(best_model, list_of_weights)
        # This group requires finding the closest model to the returned one
        # (weights distance wise)
        elif aggregation_rule in byzantine_resilient_many_return_models_aggregation:
            # when different aggregation strategies available
            # write a function to find the closest model
            raise NotImplementedError(
                "aggregate_bulyan currently does not support the aggregation rules that"
                " return many models as results. "
                "Such aggregation rules are currently not available in Flower."
            )
        else:
            raise ValueError(
                "The given aggregation rule is not added as Byzantine resilient. "
                "Please choose from Byzantine resilient rules."
            )

        selected_models_set.append(results[best_idx])

        # remove idx from tracker and weights_results
        results.pop(best_idx)

    # Compute median parameter vector across selected_models_set
    median_vect = aggregate_median(selected_models_set)

    # Take the averaged beta parameters of the closest distance to the median
    # (coordinate-wise)
    parameters_aggregated = _aggregate_n_closest_weights(
        median_vect, selected_models_set, beta_closest=beta
    )
    return parameters_aggregated


def weighted_loss_avg(results: List[Tuple[int, float]]) -> float:
    """Aggregate evaluation results obtained from multiple clients."""
    num_total_evaluation_examples = sum(num_examples for (num_examples, _) in results)
    weighted_losses = [num_examples * loss for num_examples, loss in results]
    return sum(weighted_losses) / num_total_evaluation_examples


def aggregate_qffl(
    parameters: NDArrays, deltas: List[NDArrays], hs_fll: List[NDArrays]
) -> NDArrays:
    """Compute weighted average based on Q-FFL paper."""
    demominator: float = np.sum(np.asarray(hs_fll))
    scaled_deltas = []
    for client_delta in deltas:
        scaled_deltas.append([layer * 1.0 / demominator for layer in client_delta])
    updates = []
    for i in range(len(deltas[0])):
        tmp = scaled_deltas[0][i]
        for j in range(1, len(deltas)):
            tmp += scaled_deltas[j][i]
        updates.append(tmp)
    new_parameters = [(u - v) * 1.0 for u, v in zip(parameters, updates)]
    return new_parameters


def _compute_distances(weights: List[NDArrays]) -> NDArray:
    """Compute distances between vectors.

    Input: weights - list of weights vectors
    Output: distances - matrix distance_matrix of squared distances between the vectors
    """
    flat_w = np.array([np.concatenate(p, axis=None).ravel() for p in weights])
    distance_matrix = np.zeros((len(weights), len(weights)))
    for i, flat_w_i in enumerate(flat_w):
        for j, flat_w_j in enumerate(flat_w):
            delta = flat_w_i - flat_w_j
            norm = np.linalg.norm(delta)
            distance_matrix[i, j] = norm**2
    return distance_matrix


def _trim_mean(array: NDArray, proportiontocut: float) -> NDArray:
    """Compute trimmed mean along axis=0.

    It is based on the scipy implementation.

    https://docs.scipy.org/doc/scipy/reference/generated/
    scipy.stats.trim_mean.html.
    """
    axis = 0
    nobs = array.shape[axis]
    lowercut = int(proportiontocut * nobs)
    uppercut = nobs - lowercut
    if lowercut > uppercut:
        raise ValueError("Proportion too big.")

    atmp = np.partition(array, (lowercut, uppercut - 1), axis)

    slice_list = [slice(None)] * atmp.ndim
    slice_list[axis] = slice(lowercut, uppercut)
    result: NDArray = np.mean(atmp[tuple(slice_list)], axis=axis)
    return result


def aggregate_trimmed_avg(
    results: List[Tuple[NDArrays, int]], proportiontocut: float
) -> NDArrays:
    """Compute trimmed average."""
    # Create a list of weights and ignore the number of examples
    weights = [weights for weights, _ in results]

    trimmed_w: NDArrays = [
        _trim_mean(np.asarray(layer), proportiontocut=proportiontocut)
        for layer in zip(*weights)
    ]

    return trimmed_w


def _check_weights_equality(weights1: NDArrays, weights2: NDArrays) -> bool:
    """Check if weights are the same."""
    if len(weights1) != len(weights2):
        return False
    return all(
        np.array_equal(layer_weights1, layer_weights2)
        for layer_weights1, layer_weights2 in zip(weights1, weights2)
    )


def _find_reference_weights(
    reference_weights: NDArrays, list_of_weights: List[NDArrays]
) -> int:
    """Find the reference weights by looping through the `list_of_weights`.

    Raise Error if the reference weights is not found.

    Parameters
    ----------
    reference_weights: NDArrays
        Weights that will be searched for.
    list_of_weights: List[NDArrays]
        List of weights that will be searched through.

    Returns
    -------
    index: int
        The index of `reference_weights` in the `list_of_weights`.

    Raises
    ------
    ValueError
        If `reference_weights` is not found in `list_of_weights`.
    """
    for idx, weights in enumerate(list_of_weights):
        if _check_weights_equality(reference_weights, weights):
            return idx
    raise ValueError("The reference weights not found in list_of_weights.")


def _aggregate_n_closest_weights(
    reference_weights: NDArrays, results: List[Tuple[NDArrays, int]], beta_closest: int
) -> NDArrays:
    """Calculate element-wise mean of the `N` closest values.

    Note, each i-th coordinate of the result weight is the average of the beta_closest
    -ith coordinates to the reference weights


    Parameters
    ----------
    reference_weights: NDArrays
        The weights from which the distances will be computed
    results: List[Tuple[NDArrays, int]]
        The weights from models
    beta_closest: int
        The number of the closest distance weights that will be averaged

    Returns
    -------
    aggregated_weights: NDArrays
        Averaged (element-wise) beta weights that have the closest distance to
         reference weights
    """
    list_of_weights = [weights for weights, num_examples in results]
    aggregated_weights = []

    for layer_id, layer_weights in enumerate(reference_weights):
        other_weights_layer_list = []
        for other_w in list_of_weights:
            other_weights_layer = other_w[layer_id]
            other_weights_layer_list.append(other_weights_layer)
        other_weights_layer_np = np.array(other_weights_layer_list)
        diff_np = np.abs(layer_weights - other_weights_layer_np)
        # Create indices of the smallest differences
        # We do not need the exact order but just the beta closest weights
        # therefore np.argpartition is used instead of np.argsort
        indices = np.argpartition(diff_np, kth=beta_closest - 1, axis=0)
        # Take the weights (coordinate-wise) corresponding to the beta of the
        # closest distances
        beta_closest_weights = np.take_along_axis(
            other_weights_layer_np, indices=indices, axis=0
        )[:beta_closest]
        aggregated_weights.append(np.mean(beta_closest_weights, axis=0))
    return aggregated_weights