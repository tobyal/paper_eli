"""Validate actual cached language features and record a dataset fingerprint."""

import hashlib, json, os, time
from pathlib import Path
import numpy as np


def validate():
    root = Path(os.environ["ROOT"])
    dataset = root / "data/gsle_langsplat"
    names = sorted(path.stem for path in (dataset / "images").iterdir())
    assert len(names) == 306
    digest = hashlib.sha256()
    counts = []
    minimum = 2.0
    maximum = 0.0
    start = time.time()
    for name in names:
        feature_path = dataset / "language_features" / (name + "_f.npy")
        segment_path = dataset / "language_features" / (name + "_s.npy")
        features = np.load(feature_path)
        segments = np.load(segment_path)
        assert (
            features.ndim == 2
            and features.shape[1] == 512
            and features.dtype == np.float32
        ), (name, features.shape, features.dtype)
        assert segments.shape == (4, 528, 938) and segments.dtype == np.int32, (
            name,
            segments.shape,
            segments.dtype,
        )
        assert segments.min() >= -1 and segments.max() < len(features), name
        assert np.isfinite(features).all(), name
        norms = np.linalg.norm(features, axis=1)
        assert norms.min() > 0.99 and norms.max() < 1.01, (
            name,
            float(norms.min()),
            float(norms.max()),
        )
        minimum = min(minimum, float(norms.min()))
        maximum = max(maximum, float(norms.max()))
        counts.append(len(features))
        for path in (feature_path, segment_path):
            digest.update(path.name.encode())
            digest.update(path.read_bytes())
    record = {
        "views": len(names),
        "feature_vectors": sum(counts),
        "feature_dimension": 512,
        "segment_shape": [4, 528, 938],
        "all_finite": True,
        "all_segment_indices_valid": True,
        "embedding_norm_range": [minimum, maximum],
        "ordered_filename_and_file_bytes_sha256": digest.hexdigest(),
        "seconds": time.time() - start,
    }
    (root / "records/gsle/feature_cache_validation.json").write_text(
        json.dumps(record, indent=2)
    )
    print(record, flush=True)
    return record


if __name__ == "__main__":
    validate()
