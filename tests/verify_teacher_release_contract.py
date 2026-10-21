"""Learning-free checks for teacher fusion and experiment identity."""

import os
import sys
import tempfile
import unittest

import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
for path in (os.path.join(ROOT, "model"), os.path.join(ROOT, "vild"),
             os.path.join(ROOT, "utils"), os.path.join(ROOT, "shared_vild")):
    if path not in sys.path:
        sys.path.insert(0, path)

from experiment_identity import checkpoint_experiment_id
from teacher_fusion import WeightedTeacherFusion


class TeacherReleaseContractTest(unittest.TestCase):
    def test_feature_fusion_is_invariant_to_positive_feature_scale(self):
        outputs = {
            "a": {"logits": torch.tensor([[2.0, 0.0]]), "features": torch.tensor([[1.0, 2.0, 0.0]])},
            "b": {"logits": torch.tensor([[1.0, 0.0]]), "features": torch.tensor([[0.0, 1.0, 3.0]])},
        }
        class_map = {"a": 0, "b": 1, "others": 2}
        original = WeightedTeacherFusion(class_map, 3, "cpu", score_mode="margin").fuse(outputs)
        scaled = {name: {"logits": value["logits"], "features": value["features"].clone()}
                  for name, value in outputs.items()}
        scaled["a"]["features"] *= 100.0
        changed = WeightedTeacherFusion(class_map, 3, "cpu", score_mode="margin").fuse(scaled)
        self.assertTrue(torch.allclose(original.features, changed.features, atol=1e-7))
        self.assertTrue(torch.equal(original.logits, changed.logits))

    def test_checkpoint_experiment_id_is_read_back(self):
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "checkpoint.pth")
            torch.save({"config_metadata": {"experiment_id": "expected"}}, path)
            self.assertEqual(checkpoint_experiment_id(path), "expected")


if __name__ == "__main__":
    unittest.main()

