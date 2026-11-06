"""Check raw-category round trips and probability normalization."""

import hashlib
import json
from pathlib import Path
import numpy as np


def clean_data(core):
    raise RuntimeError("Supply the private workbook through launch.py")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def digest(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def write(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def probability_tolerance(native_dtype):
    if not np.issubdtype(native_dtype, np.floating):
        raise TypeError("Expected floating-point probability output")
    return float(max(1e-12, 8 * np.finfo(native_dtype).eps))


def roundtrip(core, estimator, encoded, train, evaluation, schema, probability):
    codec = core.RawFeatureCodec.fit(train, schema)
    transformed = core.transform_raw(encoded.preprocessor, codec.decode(codec.encode(evaluation)))
    np.testing.assert_array_equal(transformed, encoded.X_test)
    actual = core.predict_proba_aligned(estimator, transformed, np.array([0, 1, 2]))
    np.testing.assert_allclose(actual, probability, rtol=0, atol=1e-12)
    assert probability.shape == (len(evaluation), 3) and np.isfinite(probability).all()
    native_dtype = np.asarray(estimator.predict_proba(encoded.X_test[:1])).dtype
    normalization_tolerance = probability_tolerance(native_dtype)
    np.testing.assert_allclose(probability.sum(axis=1), 1, rtol=0, atol=normalization_tolerance)
    assert ((probability >= -1e-12) & (probability <= 1 + 1e-12)).all()
    unknown = {
        f: int((~evaluation[f].isin(train[f].unique())).sum()) for f in schema.categorical_cols
    }
    return dict(
        max_probability_error=float(np.max(abs(actual - probability))),
        native_probability_dtype=str(native_dtype),
        sum_to_one_max_error=float(abs(probability.sum(axis=1) - 1).max()),
        sum_to_one_tolerance=normalization_tolerance,
        unknown_category_rows={f: n for f, n in unknown.items() if n},
        encoded_columns=int(encoded.X_test.shape[1]),
    )
