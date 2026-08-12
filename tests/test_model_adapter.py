"""Tests for the stand-in predictor and its attributions."""

from __future__ import annotations

import numpy as np
import pytest

from app.model_adapter import LinearStandInPredictor, verify_against_shap


@pytest.fixture(scope="module")
def predictor() -> LinearStandInPredictor:
    return LinearStandInPredictor()


def test_predicts_one_of_the_known_classes(predictor):
    result = predictor.predict("Forest fire near La Ronge Sask. Canada")
    assert result.predicted_class in predictor.class_names
    assert 0.5 <= result.confidence <= 1.0


def test_probabilities_sum_to_one(predictor):
    result = predictor.predict("Roads blocked but we're safe, thanks.")
    assert sum(result.class_probabilities.values()) == pytest.approx(1.0, abs=1e-6)


def test_segments_reconstruct_the_input_exactly(predictor):
    text = "  Power out on Elm St.  We need help!! \n(please)  "
    result = predictor.predict(text)
    assert "".join(segment.text for segment in result.segments) == text


def test_attribution_decomposition_is_additive(predictor):
    """base + absent + sum(token contributions) must equal the logit."""
    text = "My house collapsed, trapped under beams"
    result = predictor.predict(text)
    token_total = sum(s.contribution for s in result.segments if s.is_token)
    reconstructed = result.base_value + result.absent_contribution + token_total
    assert reconstructed == pytest.approx(result.logit, abs=1e-4)


def test_closed_form_matches_shap_linear_explainer(predictor):
    """Our analytic SHAP values agree with the shap library's."""
    largest_gap = verify_against_shap(predictor, "Forest fire near La Ronge Sask. Canada")
    assert largest_gap < 1e-8


def test_unknown_words_get_zero_contribution(predictor):
    result = predictor.predict("zzzqqxvv wubbalubba")
    token_scores = [s.contribution for s in result.segments if s.is_token]
    assert all(score == 0.0 for score in token_scores)


def test_disaster_wording_scores_higher_than_small_talk(predictor):
    disaster = predictor.predict("Massive earthquake, buildings collapsed, people trapped")
    chatter = predictor.predict("I love fruits and sunny afternoons")
    assert disaster.logit > chatter.logit


def test_empty_tokens_do_not_crash(predictor):
    result = predictor.predict("!!! ???")
    assert result.segments
    assert np.isfinite(result.logit)
