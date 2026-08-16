"""
Tests for End-to-End Pipeline
"""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

import config
from schemas import MeetingInput, PipelineOutput
from pipeline import LISTENPipeline


# ── Test Fixtures ──

@pytest.fixture(scope="module")
def sample_meeting_json():
    """Load the sample meeting JSON."""
    json_path = Path(__file__).parent.parent / "data" / "sample_meeting.json"
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)


@pytest.fixture(scope="module")
def pipeline():
    """Create pipeline without LLM (for fast testing)."""
    return LISTENPipeline(gat_checkpoint=None, use_llm=False)


# ── Pipeline Tests ──

class TestPipeline:
    """Test the end-to-end pipeline."""

    def test_pipeline_runs(self, pipeline, sample_meeting_json):
        """Pipeline should run without errors on sample data."""
        result = pipeline.run(
            meeting_input=sample_meeting_json,
            question="Why was rubber chosen for the remote control?",
        )
        assert isinstance(result, PipelineOutput)

    def test_output_has_answer(self, pipeline, sample_meeting_json):
        """Output should contain a non-empty answer."""
        result = pipeline.run(
            meeting_input=sample_meeting_json,
            question="What material was chosen?",
        )
        assert result.answer, "Answer should not be empty"
        assert len(result.answer) > 10, "Answer should be substantive"

    def test_output_has_evidence_graph(self, pipeline, sample_meeting_json):
        """Output should contain an evidence graph with nodes and edges."""
        result = pipeline.run(
            meeting_input=sample_meeting_json,
            question="What was decided about the remote?",
        )
        assert len(result.evidence_graph.nodes) > 0, "Should have evidence nodes"
        assert len(result.top_segments) > 0, "Should have top segments"

    def test_node_weights_valid(self, pipeline, sample_meeting_json):
        """All node weights should be in [0, 1]."""
        result = pipeline.run(
            meeting_input=sample_meeting_json,
            question="What was the cost comparison?",
        )
        for node in result.evidence_graph.nodes:
            assert 0.0 <= node.weight <= 1.0, f"Invalid weight {node.weight} for {node.id}"

    def test_top_segments_ordered(self, pipeline, sample_meeting_json):
        """Top segments should match node ordering by weight."""
        result = pipeline.run(
            meeting_input=sample_meeting_json,
            question="What material?",
        )
        assert result.top_segments[0] == result.evidence_graph.nodes[0].id

    def test_output_serializable(self, pipeline, sample_meeting_json):
        """Output should be JSON-serializable (for Person C)."""
        result = pipeline.run(
            meeting_input=sample_meeting_json,
            question="What was discussed?",
        )
        output_dict = result.model_dump()
        json_str = json.dumps(output_dict, ensure_ascii=False)
        assert len(json_str) > 100, "JSON output too short"

        # Verify it can be parsed back
        parsed = json.loads(json_str)
        assert "answer" in parsed
        assert "evidence_graph" in parsed
        assert "nodes" in parsed["evidence_graph"]

    def test_metadata_populated(self, pipeline, sample_meeting_json):
        """Model metadata should be populated."""
        result = pipeline.run(
            meeting_input=sample_meeting_json,
            question="What was decided?",
        )
        assert result.model_metadata is not None
        assert result.model_metadata.num_segments_retrieved > 0
        assert result.model_metadata.num_segments_in_graph > 0

    def test_counterfactual_runs(self, pipeline, sample_meeting_json):
        """Counterfactual re-run should produce a different result."""
        # Get original result
        original = pipeline.run(
            meeting_input=sample_meeting_json,
            question="Why was rubber chosen?",
        )

        # Remove the top segment and re-run
        if original.top_segments:
            counterfactual = pipeline.run_counterfactual(
                meeting_input=sample_meeting_json,
                question="Why was rubber chosen?",
                remove_segment_ids=[original.top_segments[0]],
            )
            assert isinstance(counterfactual, PipelineOutput)
            # The removed segment should not be in the counterfactual results
            cf_ids = [n.id for n in counterfactual.evidence_graph.nodes]
            assert original.top_segments[0] not in cf_ids

    def test_input_validation(self, pipeline):
        """Pipeline should validate input schema."""
        with pytest.raises(Exception):
            pipeline.run(
                meeting_input={"invalid": "data"},
                question="test",
            )

    def test_custom_top_k(self, pipeline, sample_meeting_json):
        """Custom top_k should override defaults."""
        result = pipeline.run(
            meeting_input=sample_meeting_json,
            question="What was discussed?",
            top_k_retrieval=5,
        )
        assert len(result.evidence_graph.nodes) <= 5


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
