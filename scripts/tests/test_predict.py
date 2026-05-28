"""Tests for scripts/predict.py — deterministic prediction functions."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from predict import winprob, confidence_bucket


def test_winprob_known_points():
    assert abs(winprob(0.0) - 0.500) < 0.005
    assert abs(winprob(0.81) - 0.580) < 0.005
    assert abs(winprob(1.76) - 0.670) < 0.005
    assert abs(winprob(0.30) - 0.530) < 0.005
    assert abs(winprob(-1.0) - (1 - winprob(1.0))) < 1e-9


def test_confidence_bucket_boundaries():
    assert confidence_bucket(0.55) == "LOW"
    assert confidence_bucket(0.579) == "LOW"
    assert confidence_bucket(0.58) == "MEDIUM"
    assert confidence_bucket(0.669) == "MEDIUM"
    assert confidence_bucket(0.67) == "HIGH"
    assert confidence_bucket(0.80) == "HIGH"
