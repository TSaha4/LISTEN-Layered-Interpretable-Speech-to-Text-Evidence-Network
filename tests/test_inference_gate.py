"""GATInference must load a fitted ReliabilityGate stored in the checkpoint."""

from __future__ import annotations

import torch

from app.dl.gat_model import EvidenceGAT, ReliabilityGate
from app.dl.inference import GATInference


def test_gate_state_dict_is_loaded_from_checkpoint(tmp_path) -> None:
    torch.manual_seed(0)
    ckpt = tmp_path / "gat.pt"
    torch.save(
        {"model_state_dict": EvidenceGAT().state_dict(), "gate_state_dict": ReliabilityGate(0.3, 0.2).state_dict()},
        ckpt,
    )
    inference = GATInference(checkpoint_path=ckpt)
    assert inference.gate.node_beta.item() == pytest_approx(0.3)
    assert inference.gate.edge_beta.item() == pytest_approx(0.2)


def test_checkpoint_without_gate_uses_config_defaults(tmp_path) -> None:
    from app import config

    ckpt = tmp_path / "gat.pt"
    torch.save({"model_state_dict": EvidenceGAT().state_dict()}, ckpt)
    inference = GATInference(checkpoint_path=ckpt)
    assert inference.gate.node_beta.item() == pytest_approx(config.RELIABILITY_GATE_BETA)


def pytest_approx(value: float):
    import pytest

    return pytest.approx(value, abs=1e-6)
