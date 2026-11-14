"""CPU-only failure-injection tests; no research audio or trained model is used.

Run from the package directory with:
    python -m pytest -q tests/test_resume_integrity.py

The actual train_one/checkpoint/completion implementation is exercised with a
tiny artificial model, artificial waves, and a temporary placeholder weights
file. All generated artifacts are confined to pytest's temporary directory.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

BASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE / "src"))
import feedback_core as f


class TinyStudent(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.dropout = torch.nn.Dropout(0.2)
        self.linear = torch.nn.Linear(1, 9)

    def forward(self, wave, augment=False):
        return self.linear(self.dropout(wave))


def artificial_frame(n):
    return pd.DataFrame(
        [
            dict(
                path=f"artificial_{i}.wav",
                filename=f"artificial_{i}.wav",
                label=f.CLASSES[i % 9],
                group=f"artificial_group_{i}",
                sha256=f"{i:064x}",
            )
            for i in range(n)
        ]
    )


@pytest.fixture
def harness(tmp_path, monkeypatch):
    # Force the real CPU branch even when this test runs in a GPU notebook.
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(f, "build_model", lambda *args, **kwargs: TinyStudent())
    monkeypatch.setattr(
        f,
        "cached_waves",
        lambda frame, *args, **kwargs: torch.arange(
            len(frame), dtype=torch.float32
        ).reshape(-1, 1)
        / 100,
    )
    # Freeze the source fingerprint for this test and avoid dependencies on a
    # particular D-drive checkout or real pretrained weights.
    snapshot = tmp_path / "source_snapshot"
    snapshot.mkdir()
    (snapshot / "feedback_core.py").write_bytes(Path(f.__file__).read_bytes())
    (snapshot / "provenance_active.csv").write_text("synthetic fixture\n")
    monkeypatch.setattr(f, "HERE", snapshot)
    root = tmp_path / "project"
    (root / "weights").mkdir(parents=True)
    (root / "weights" / "mn10_as_mAP_471.pt").write_bytes(
        b"placeholder used only for the run fingerprint"
    )
    train, val, test = artificial_frame(32), artificial_frame(9), artificial_frame(9)
    targets = dict(
        q=np.zeros((32, 9), dtype=np.float32),
        first=np.zeros(32, dtype=np.int64),
        second=np.full(32, -1, dtype=np.int64),
        weight=np.zeros(32, dtype=np.float32),
    )
    recipe = dict(stage="1_3", n=200, seed=1, variant="supervised", split_seed=0)
    old_threads = torch.get_num_threads()
    torch.set_num_threads(1)

    def run(folder, epochs=2):
        f.train_one(
            root,
            folder,
            train,
            val,
            test,
            targets,
            recipe,
            dict(epochs=epochs),
            {},
        )

    yield run
    torch.set_num_threads(old_threads)


@pytest.mark.parametrize("interrupted_epoch", [0, 1], ids=["mid_run", "final_epoch"])
def test_resume_after_checkpoint_commit_matches_uninterrupted(
    harness, tmp_path, monkeypatch, interrupted_epoch
):
    normal = tmp_path / "uninterrupted"
    interrupted = tmp_path / "interrupted"
    harness(normal)
    expected = torch.load(normal / "last.pt", map_location="cpu", weights_only=False)
    save = f._atomic_torch_save

    def disconnect_after_commit(path, obj):
        save(path, obj)
        if Path(path).name == "last.pt" and obj["epoch"] == interrupted_epoch:
            raise InterruptedError("artificial disconnection after checkpoint commit")

    monkeypatch.setattr(f, "_atomic_torch_save", disconnect_after_commit)
    with pytest.raises(InterruptedError, match="artificial disconnection"):
        harness(interrupted)
    monkeypatch.setattr(f, "_atomic_torch_save", save)
    harness(interrupted)
    actual = torch.load(interrupted / "last.pt", map_location="cpu", weights_only=False)
    for key, value in expected["model"].items():
        torch.testing.assert_close(actual["model"][key], value, rtol=0, atol=0)
    for key, value in expected["best_model"].items():
        torch.testing.assert_close(actual["best_model"][key], value, rtol=0, atol=0)
    assert actual["best_epoch"] == expected["best_epoch"]
    assert actual["best"] == expected["best"]
    pd.testing.assert_frame_equal(
        pd.read_csv(normal / "history.csv").drop(columns="seconds"),
        pd.read_csv(interrupted / "history.csv").drop(columns="seconds"),
        check_exact=True,
    )
    assert len(pd.read_csv(interrupted / "history.csv")) == 2
    assert f.validate_complete(interrupted)


def test_completed_run_skips_and_each_artifact_is_verified(harness, tmp_path, monkeypatch):
    folder = tmp_path / "completed"
    harness(folder, epochs=1)
    protocol = json.loads((folder / "protocol.json").read_text())
    assert protocol["runtime"]["device"] == "cpu"

    def forbid_model_initialization(*args, **kwargs):
        raise AssertionError("completed run attempted model initialization")

    monkeypatch.setattr(f, "build_model", forbid_model_initialization)
    harness(folder, epochs=1)
    for artifact in f.COMPLETE_FILES:
        path = folder / artifact
        original = path.read_bytes()
        path.write_bytes(original + b" ")
        try:
            with pytest.raises(RuntimeError, match="artifact damaged"):
                f.validate_complete(folder)
        finally:
            path.write_bytes(original)
    assert f.validate_complete(folder)


def test_runtime_change_cannot_reuse_completed_run(harness, tmp_path, monkeypatch):
    folder = tmp_path / "completed"
    harness(folder, epochs=1)
    changed = dict(f.runtime_metadata(), torch="artificial different torch version")
    monkeypatch.setattr(f, "runtime_metadata", lambda: changed)
    with pytest.raises(RuntimeError, match="Run settings changed"):
        harness(folder, epochs=1)
