import numpy as np

from polarfunc_repro.ood_baselines import (
    energy_score,
    fit_distance_state,
    mahalanobis_score,
    maxlogit_score,
    msp_score,
    prototype_cosine_score,
)


def test_logit_scores_rank_low_confidence_example_as_more_ood():
    logits = np.array([[8.0, -2.0], [0.1, 0.0]])
    for scorer in (msp_score, maxlogit_score, energy_score):
        scores = scorer(logits)
        assert scores[1] > scores[0]


def test_distance_scores_rank_remote_example_as_more_ood():
    x_train = np.array([[0.0, 0.0], [0.1, 0.0], [3.0, 3.0], [3.1, 3.0]])
    y_train = np.array(["a", "a", "b", "b"])
    state = fit_distance_state(x_train, y_train, variance_floor=0.01)
    queries = np.array([[0.05, 0.02], [20.0, -20.0]])
    assert mahalanobis_score(queries, state)[1] > mahalanobis_score(queries, state)[0]
    assert prototype_cosine_score(queries, state)[1] > prototype_cosine_score(queries, state)[0]
