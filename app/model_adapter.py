"""
Model-loading seam for the demo.

Everything the API needs from a model is expressed by `Predictor`. Today the
only implementation is `LinearStandInPredictor`, which wraps the TF-IDF +
logistic-regression stand-in. If the original fine-tuned BERT checkpoint ever
turns up, add a `BertPredictor` here that satisfies the same protocol and the
API and frontend need no changes.

On the explanations
-------------------
For a linear model the SHAP values have a closed form:

    phi_j = w_j * (x_j - E[x_j])          base = w . E[x] + b

and `sum_j phi_j + base == logit(x)` exactly. There is no sampling and no
approximation, so `shap.LinearExplainer` would return these same numbers -- it
is used when installed purely as a cross-check (see `verify_against_shap`).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

import joblib
import numpy as np

ARTIFACT_PATH = Path(__file__).resolve().parent.parent / "artifacts" / "standin_model.joblib"


@dataclass
class Segment:
    """One piece of the original text, in order. Concatenating every `text`
    reproduces the input exactly, so the frontend can render without guessing
    at whitespace."""

    text: str
    contribution: float = 0.0
    is_token: bool = False


@dataclass
class Prediction:
    predicted_class: str
    predicted_index: int
    confidence: float
    class_probabilities: dict[str, float]
    segments: list[Segment] = field(default_factory=list)
    # Additive decomposition, exposed so the numbers can be audited:
    #   base_value + absent_contribution + sum(token contributions) == logit
    base_value: float = 0.0
    absent_contribution: float = 0.0
    logit: float = 0.0
    model_kind: str = "unknown"
    is_standin: bool = True


class Predictor(Protocol):
    def predict(self, text: str) -> Prediction: ...

    @property
    def class_names(self) -> list[str]: ...


class LinearStandInPredictor:
    """TF-IDF + logistic regression with exact per-word SHAP attributions."""

    def __init__(self, artifact_path: Path = ARTIFACT_PATH) -> None:
        if not artifact_path.exists():
            raise FileNotFoundError(
                f"No trained stand-in model at {artifact_path}. "
                "Run `python training/train_standin.py` first."
            )
        bundle = joblib.load(artifact_path)
        self._pipeline = bundle["pipeline"]
        self._class_names: list[str] = bundle["class_names"]
        self._feature_means: np.ndarray = bundle["feature_means"]
        self._token_re = re.compile(bundle["token_pattern"])
        self.metrics: dict = bundle.get("metrics", {})

        self._vectorizer = self._pipeline.named_steps["tfidf"]
        self._classifier = self._pipeline.named_steps["clf"]
        self._feature_names = self._vectorizer.get_feature_names_out()
        # Binary logistic regression: one weight row, positive class at index 1.
        self._weights = self._classifier.coef_[0]
        self._intercept = float(self._classifier.intercept_[0])
        self._preprocess = self._vectorizer.build_preprocessor()

        # Map each vocabulary term to the tokens it is built from, once, so
        # per-request work is proportional to the input rather than the vocab.
        self._term_parts = {
            name: tuple(name.split(" ")) for name in self._feature_names
        }

    @property
    def class_names(self) -> list[str]:
        return list(self._class_names)

    def _tokenize_with_spans(self, text: str) -> list[tuple[int, int, str]]:
        """Token spans over the *raw* text, each paired with its normalized form.

        Spans come from the raw string so they line up with what the user typed.
        Normalization is applied per-token rather than to the whole document;
        for lowercasing and accent-stripping this yields the same tokens the
        vectorizer's analyzer produces, without shifting any offsets.
        """
        spans = []
        for match in self._token_re.finditer(text):
            normalized = self._preprocess(match.group())
            spans.append((match.start(), match.end(), normalized))
        return spans

    def _attribute(self, text: str, row: np.ndarray) -> tuple[list[Segment], float]:
        """Split each active feature's SHAP value across the raw-text tokens
        that produced it, then stitch the text back together as segments."""
        spans = self._tokenize_with_spans(text)
        token_scores = np.zeros(len(spans))

        normalized_tokens = [normalized for _, _, normalized in spans]
        # position lookup for unigrams, and for bigram left-halves
        positions: dict[str, list[int]] = {}
        for index, token in enumerate(normalized_tokens):
            positions.setdefault(token, []).append(index)

        active = row.nonzero()[0]
        attributed_total = 0.0

        for feature_index in active:
            phi = float(self._weights[feature_index] * (row[feature_index] - self._feature_means[feature_index]))
            if phi == 0.0:
                continue
            parts = self._term_parts[self._feature_names[feature_index]]

            if len(parts) == 1:
                targets = [[index] for index in positions.get(parts[0], [])]
            else:
                # bigram: every adjacent token pair that matches
                targets = [
                    list(range(start, start + len(parts)))
                    for start in positions.get(parts[0], [])
                    if start + len(parts) <= len(normalized_tokens)
                    and tuple(normalized_tokens[start : start + len(parts)]) == parts
                ]

            if not targets:
                # Feature fired but its tokens are not locatable in the raw text
                # (rare: accent-stripping edge cases). Keep it out of the token
                # scores rather than misattributing it; it stays in `logit`.
                continue

            share = phi / sum(len(group) for group in targets)
            for group in targets:
                for index in group:
                    token_scores[index] += share
            attributed_total += phi

        segments: list[Segment] = []
        cursor = 0
        for (start, end, _), score in zip(spans, token_scores):
            if start > cursor:
                segments.append(Segment(text=text[cursor:start]))
            segments.append(Segment(text=text[start:end], contribution=round(score, 6), is_token=True))
            cursor = end
        if cursor < len(text):
            segments.append(Segment(text=text[cursor:]))

        return segments, attributed_total

    def predict(self, text: str) -> Prediction:
        row = np.asarray(self._vectorizer.transform([text]).todense()).ravel()

        logit = float(self._weights @ row + self._intercept)
        base_value = float(self._weights @ self._feature_means + self._intercept)
        probability_positive = 1.0 / (1.0 + np.exp(-logit))

        segments, attributed = self._attribute(text, row)

        predicted_index = int(probability_positive >= 0.5)
        probabilities = {
            self._class_names[0]: round(float(1.0 - probability_positive), 6),
            self._class_names[1]: round(float(probability_positive), 6),
        }

        return Prediction(
            predicted_class=self._class_names[predicted_index],
            predicted_index=predicted_index,
            confidence=round(max(probabilities.values()), 6),
            class_probabilities=probabilities,
            segments=segments,
            base_value=round(base_value, 6),
            # Everything not pinned to a visible token: features absent from the
            # text plus any unlocatable term. Reported rather than hidden so the
            # decomposition still adds up to `logit`.
            absent_contribution=round(logit - base_value - attributed, 6),
            logit=round(logit, 6),
            model_kind="tfidf+logreg (stand-in)",
            is_standin=True,
        )


def verify_against_shap(predictor: LinearStandInPredictor, text: str) -> float:
    """Max absolute difference between our closed-form values and
    `shap.LinearExplainer`. Returns 0.0-ish when they agree. Test-only helper;
    `shap` is not needed to serve predictions."""
    import shap  # imported lazily so serving never depends on it

    row = np.asarray(predictor._vectorizer.transform([text]).todense())
    explainer = shap.LinearExplainer(
        predictor._classifier,
        masker=shap.maskers.Independent(predictor._feature_means.reshape(1, -1)),
    )
    reference = explainer.shap_values(row)[0]
    ours = predictor._weights * (row.ravel() - predictor._feature_means)
    return float(np.max(np.abs(reference - ours)))
