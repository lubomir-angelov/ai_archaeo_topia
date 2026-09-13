#!/usr/bin/env python3
"""Tests for the per-epoch ``metrics.json`` record.

Per-sample prediction rows belong in the ``prediction_stats_*.jsonl`` files,
not in ``metrics.json``.  Embedding them made a 200-epoch run's metrics file
19 MB, of which 12.9 MB was the validation rows, all of it a second copy of
data already written to disk.
"""

from __future__ import annotations

from typing import Any

from archeo_topia.training.train_mapsam_v0 import build_metrics_record


def _metrics(n_rows: int) -> dict[str, Any]:
    return {
        "loss": 0.25,
        "bce_loss": 0.05,
        "dice_loss": 0.20,
        "mean_iou": 0.68,
        "mean_dice": 0.80,
        "prediction_stats": [{"sample_id": f"s{i}", "iou": 0.5} for i in range(n_rows)],
        "prediction_stats_aggregate": {"mean_iou": 0.68},
    }


class TestBuildMetricsRecord:
    def test_per_sample_rows_are_dropped_from_val(self) -> None:
        rec = build_metrics_record(5, _metrics(3), _metrics(1000))
        assert "prediction_stats" not in rec["val"]

    def test_per_sample_rows_are_dropped_from_train(self) -> None:
        rec = build_metrics_record(5, _metrics(3), _metrics(3))
        assert "prediction_stats" not in rec["train"]

    def test_aggregates_are_kept(self) -> None:
        rec = build_metrics_record(5, _metrics(3), _metrics(3))
        assert rec["train"]["prediction_stats_aggregate"] == {"mean_iou": 0.68}
        assert rec["val"]["prediction_stats_aggregate"] == {"mean_iou": 0.68}

    def test_scalar_metrics_are_kept(self) -> None:
        rec = build_metrics_record(7, _metrics(3), _metrics(3))
        assert rec["epoch"] == 7
        for key in ("loss", "bce_loss", "dice_loss", "mean_iou", "mean_dice"):
            assert rec["val"][key] == _metrics(3)[key]

    def test_unvalidated_epoch_keeps_none(self) -> None:
        rec = build_metrics_record(3, _metrics(3), None)
        assert rec["val"] is None

    def test_record_stays_small_for_many_rows(self) -> None:
        """The record size must not scale with the number of samples."""
        import json

        small = len(json.dumps(build_metrics_record(1, _metrics(5), _metrics(5))))
        large = len(json.dumps(build_metrics_record(1, _metrics(5000), _metrics(5000))))
        assert small == large

    def test_source_metrics_are_not_mutated(self) -> None:
        """The caller still needs prediction_stats to write the jsonl files."""
        train, val = _metrics(3), _metrics(3)
        build_metrics_record(1, train, val)
        assert "prediction_stats" in train
        assert "prediction_stats" in val
