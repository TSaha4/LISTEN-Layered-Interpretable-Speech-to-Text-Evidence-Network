"""Tests for Insights endpoints."""

from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_training_insights():
    response = client.get("/api/v1/insights/training")
    assert response.status_code == 200
    data = response.json()
    assert "available" in data
    assert "architecture" in data
    assert data["architecture"]["input_dim"] == 384
    assert data["architecture"]["num_heads"] == 4
    if data["available"]:
        assert len(data["epochs"]) > 0
        assert data["best_val_f1"] is not None
        assert data["best_val_accuracy"] is not None
        assert data["total_parameters"] is not None


def test_dataset_insights():
    response = client.get("/api/v1/insights/dataset")
    assert response.status_code == 200
    data = response.json()
    assert "available" in data
    if data["available"]:
        assert "meetings" in data
        assert "totals" in data
        assert "similarity_histogram" in data["totals"]
        assert "top_entities" in data["totals"]


def test_ablation_insights():
    response = client.get("/api/v1/insights/ablation")
    assert response.status_code == 200
    data = response.json()
    assert "available" in data
    if data["available"]:
        assert "totals" in data
        assert "queries" in data
        assert "n_queries" in data


def test_meeting_trace_404():
    response = client.get("/api/v1/meetings/non_existent_meeting_id/trace")
    assert response.status_code == 404


def test_list_meetings_and_trace():
    response = client.get("/api/v1/meetings")
    assert response.status_code == 200
    meetings = response.json()
    assert isinstance(meetings, list)
    if meetings:
        first_id = meetings[0]["meeting_id"]
        trace_res = client.get(f"/api/v1/meetings/{first_id}/trace")
        assert trace_res.status_code == 200
        trace_data = trace_res.json()
        assert "duration_s" in trace_data
        assert "n_segments" in trace_data
        assert "embedding_dim" in trace_data
        assert trace_data["embedding_dim"] == 384
