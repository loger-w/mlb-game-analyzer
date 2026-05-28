"""Tests for scripts/predict.py — deterministic prediction functions."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from predict import winprob, confidence_bucket, predict


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


def test_predict_home_favored():
    r = predict(home_score=5.5, away_score=3.0)
    assert r["direction"] == "HOME"
    assert r["total"] == 8.5
    assert abs(r["confidence_pct"] - 0.734) < 0.005
    assert r["confidence_bucket"] == "HIGH"


def test_predict_away_favored():
    r = predict(home_score=3.0, away_score=5.0)
    assert r["direction"] == "AWAY"
    assert abs(r["confidence_pct"] - 0.691) < 0.005
    assert r["confidence_bucket"] == "HIGH"


def test_predict_pickem_is_push():
    r = predict(home_score=4.1, away_score=4.0)
    assert r["direction"] == "持平"
    assert r["confidence_bucket"] is None
