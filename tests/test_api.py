"""Tests for the FastAPI surface."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app.main import STANDIN_NOTICE, app


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as test_client:
        yield test_client


def test_health_reports_loaded_model(client):
    body = client.get("/health").json()
    assert body["status"] == "ok"
    assert len(body["classes"]) == 2


def test_predict_returns_label_confidence_and_segments(client):
    response = client.post("/predict", json={"text": "Forest fire near La Ronge Sask. Canada"})
    assert response.status_code == 200
    body = response.json()
    assert body["predicted_class"] in ("disaster-related", "not disaster-related")
    assert 0.5 <= body["confidence"] <= 1.0
    assert any(segment["is_token"] for segment in body["segments"])


def test_every_prediction_carries_the_standin_notice(client):
    body = client.post("/predict", json={"text": "flood warning downtown"}).json()
    assert body["notice"] == STANDIN_NOTICE
    assert body["is_standin"] is True


def test_notice_does_not_claim_urgency_detection(client):
    """The stand-in learns disaster-relatedness, not urgency. Guard the wording
    so a future edit cannot quietly overclaim."""
    body = client.post("/predict", json={"text": "help"}).json()
    assert "not how urgent" in body["notice"]
    assert body["predicted_class"] in ("disaster-related", "not disaster-related")


def test_empty_text_is_rejected(client):
    assert client.post("/predict", json={"text": "   "}).status_code == 422


def test_overlong_text_is_rejected(client):
    assert client.post("/predict", json={"text": "x" * 2001}).status_code == 422


def test_metrics_endpoint_exposes_real_numbers(client):
    body = client.get("/metrics").json()
    assert body["is_standin"] is True
    # Guard against the paper's figures ever being pasted in here.
    assert body["accuracy"] != 0.893
    assert 0.0 < body["accuracy"] < 1.0


def test_index_page_is_served(client):
    response = client.get("/")
    assert response.status_code == 200
    assert "Crisis Text Classifier" in response.text
