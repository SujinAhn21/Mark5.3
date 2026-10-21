"""Provenance checks shared by the experiment runner and result collector."""
import hashlib
import json

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
        return (
            policy.get("classification_policy") == POLICY
            and policy.get("training_classification_policy") == POLICY
            and policy.get("run_tag") == run_tag
            and policy.get("split") == split
            and policy.get("rule") == "argmax"
            and policy.get("distill_eval_weight") == 0.0
            and policy.get("class_bias") == {}
            and policy.get("checkpoint", {}).get("sha256") == checkpoint_sha256(checkpoint_path)
        )
    except (OSError, ValueError, TypeError, AttributeError):
        return False
