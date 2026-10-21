"""Provenance checks shared by the experiment runner and result collector."""
import hashlib
import json
import os

from experiment_identity import dataset_identity

POLICY = "supervised_argmax_v1"


def checkpoint_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def matches_plain_policy(policy_path, checkpoint_path, run_tag, split):
    try:
        with open(policy_path, encoding="utf-8") as handle:
            policy = json.load(handle)
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        candidates = [
            os.path.join(project_root, f"dataset_{split}.csv"),
            os.path.join(project_root, "preprocessing", f"dataset_{split}.csv"),
        ]
        csv_path = next((path for path in candidates if os.path.isfile(path)), None)
        if csv_path is None:
            return False
        return (
            policy.get("classification_policy") == POLICY
            and policy.get("training_classification_policy") == POLICY
            and policy.get("run_tag") == run_tag
            and policy.get("split") == split
            and policy.get("rule") == "argmax"
            and policy.get("distill_eval_weight") == 0.0
            and policy.get("class_bias") == {}
            and policy.get("checkpoint", {}).get("sha256") == checkpoint_sha256(checkpoint_path)
            and policy.get("dataset_identity") == dataset_identity(csv_path)
        )
    except (OSError, ValueError, TypeError, AttributeError):
        return False
