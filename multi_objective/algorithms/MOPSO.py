# -*- coding: utf-8 -*-

import numpy as np

from multi_objective.mo_utils import (
    best_sum_history_value,
    dominates,
    evaluate_objectives,
    update_archive,
    validate_bounds,
)


def _choose_leader(archive_solutions):
    return archive_solutions[np.random.randint(len(archive_solutions))].copy()


def _update_personal_best(position, objective, best_position, best_objective):
    if dominates(objective, best_objective):
        return position.copy(), objective.copy()
    if dominates(best_objective, objective):
        return best_position, best_objective
    if np.random.rand() < 0.5:
        return position.copy(), objective.copy()
    return best_position, best_objective


def mopso(
    objective_function,
    bounds,
    swarm_size=75,
    max_iter=750,
    inertia=0.4,
    cognitive=1.5,
    social=1.5,
    archive_size=100,
    seed=None,
):
    if seed is None:
        used_seed = int(np.random.SeedSequence().generate_state(1)[0])
    else:
        used_seed = int(seed)
    np.random.seed(used_seed)

    bounds = validate_bounds(bounds)
    dimension = len(bounds)
    lower_bounds = bounds[:, 0]
    upper_bounds = bounds[:, 1]
    span = upper_bounds - lower_bounds

    positions = np.random.uniform(lower_bounds, upper_bounds, size=(swarm_size, dimension))
    velocities = np.random.uniform(-0.1 * span, 0.1 * span, size=(swarm_size, dimension))
    objectives = evaluate_objectives(objective_function, positions)
    personal_positions = positions.copy()
    personal_objectives = objectives.copy()

    archive_solutions = np.empty((0, dimension), dtype=float)
    archive_objectives = np.empty((0, objectives.shape[1]), dtype=float)
    archive_solutions, archive_objectives = update_archive(
        archive_solutions, archive_objectives, positions, objectives, archive_size
    )
    history = [best_sum_history_value(archive_objectives)]

    for _ in range(max_iter):
        for index in range(swarm_size):
            leader = _choose_leader(archive_solutions)
            r1 = np.random.rand(dimension)
            r2 = np.random.rand(dimension)
            velocities[index] = (
                inertia * velocities[index]
                + cognitive * r1 * (personal_positions[index] - positions[index])
                + social * r2 * (leader - positions[index])
            )
            velocities[index] = np.clip(velocities[index], -0.2 * span, 0.2 * span)
            positions[index] = np.clip(positions[index] + velocities[index], lower_bounds, upper_bounds)
            objectives[index] = objective_function(positions[index])
            personal_positions[index], personal_objectives[index] = _update_personal_best(
                positions[index],
                objectives[index],
                personal_positions[index],
                personal_objectives[index],
            )

        archive_solutions, archive_objectives = update_archive(
            archive_solutions, archive_objectives, positions, objectives, archive_size
        )
        history.append(best_sum_history_value(archive_objectives))

    return archive_solutions, archive_objectives, history, used_seed

