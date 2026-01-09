import json
from pathlib import Path

import pytest

from src.inference import predict_disease


def load_sample():
    p = Path(__file__).parent.parent / "sample.json"
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


def test_predict_disease_structure():
    sample = load_sample()
    result = predict_disease(sample)
    assert isinstance(result, dict)
    assert "predicted_category" in result
    assert "predicted_disease" in result
    assert "stage1_confidence" in result
    assert "stage2_confidence" in result


def test_confidences_numeric_or_none():
    sample = load_sample()
    result = predict_disease(sample)
    s1 = result.get("stage1_confidence")
    s2 = result.get("stage2_confidence")
    assert (s1 is None) or (isinstance(s1, float) and 0.0 <= s1 <= 1.0)
    assert (s2 is None) or (isinstance(s2, float) and 0.0 <= s2 <= 1.0)
