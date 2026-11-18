
from models.interface import (
    predict,
    PredictInput,
    FEATURE_ORDER,
)


def make_input():
    return PredictInput(
        schema_version="1.0",
        session_id="s1",
        window_id="w0001",
        window_started_at="2026-11-18T10:00:00Z",
        window_duration_ms=30000,
        collector_version="0.1.0",
        browser_features={
            "mouse_distance_px": 1500.0,
            "mouse_speed_mean_px_s": 160.0,
            "mouse_speed_std_px_s": 55.0,
            "direction_change_count": 8,
            "click_count": 6,
            "click_interval_mean_ms": 850.0,
            "scroll_event_count": 6,
            "scroll_distance_px": 720.0,
        },
    )


def test_output_is_simulated():
    out = predict(make_input())
    assert out.model_version == "mock-v0"


def test_prediction_placeholder():
    out = predict(make_input())
    assert out.prediction == "HUMAN"


def test_output_contains_only_model_fields():
    out = predict(make_input())
    assert set(out.model_dump().keys()) == {
        "prediction",
        "model_version",
    }


def test_feature_order_is_fixed():
    assert FEATURE_ORDER == [
        "mouse_distance_px",
        "mouse_speed_mean_px_s",
        "mouse_speed_std_px_s",
        "direction_change_count",
        "click_count",
        "click_interval_mean_ms",
        "scroll_event_count",
        "scroll_distance_px",
    ]
