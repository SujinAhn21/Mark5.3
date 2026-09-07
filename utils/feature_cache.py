import os
import sys

SHARED_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "shared_vild"))
if SHARED_DIR not in sys.path:
    sys.path.append(SHARED_DIR)

from cache_schema import get_cache_root, get_feature_cache_path, get_metadata_cache_path


def get_feature_cache_dir(project_root, mark_version):
    return get_cache_root(project_root, mark_version)
