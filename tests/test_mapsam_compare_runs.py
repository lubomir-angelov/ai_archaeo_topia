"""Tests for aggregating MapSAM runs into a comparison table."""

from __future__ import annotations

import csv
import json

from archeo_topia.analysis.mapsam_compare_runs import (
    peak_from_history,
    summarise_run,
    write_csv,
)


def _history(pairs):
    """Build a metrics history from (epoch, loss, iou) triples.

    Args:
        pairs: Iterable of ``(epoch, val_loss, val_iou)``.

    Returns:
        A history list in metrics.json shape.
    """
    return [
        {"epoch": e, "train": {"loss": 0.1}, "val": {"loss": loss, "mean_iou": iou}}
        for e, loss, iou in pairs
    ]


def _write_run(root, name, history, final_iou, config=None):
    """Write a minimal run directory.

    Args:
        root: Parent directory.
        name: Run directory name.
        history: Metrics history.
        final_iou: Final validation IoU.
        config: Optional resolved config contents.

    Returns:
        Path to the run directory.
    """
    run = root / name
    run.mkdir(parents=True)
    (run / "metrics.json").write_text(
        json.dumps(
            {
                "epochs": len(history),
                "best_val_iou": history[0]["val"]["mean_iou"] if history else 0.0,
                "final_val_iou": final_iou,
                "final_val_dice": 0.7,
                "history": history,
            }
        ),
        encoding="utf-8",
    )
    if config is not None:
        (run / "config_resolved.json").write_text(json.dumps(config), encoding="utf-8")
    return run


def _write_stats(run, rows, epoch=50):
    """Write per-sample val stats for a run.

    Args:
        run: Run directory.
        rows: Stat row dictionaries.
        epoch: Epoch number in the filename.
    """
    (run / f"prediction_stats_val_e{epoch}.jsonl").write_text(
        "\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8"
    )


def _row(sample_id, gt, pred, iou, **extra):
    """Build one per-sample stat row.

    Args:
        sample_id: Sample identifier.
        gt: Ground-truth positive pixels.
        pred: Predicted positive pixels.
        iou: IoU.
        **extra: Additional fields.

    Returns:
        A stat row dictionary.
    """
    return {
        "sample_id": sample_id,
        "gt_positive_pixels": gt,
        "pred_positive_pixels_threshold_0_5": pred,
        "iou": iou,
        "dice": 0.0,
        **extra,
    }


class TestPeakFromHistory:
    def test_finds_the_highest_iou_not_the_lowest_loss(self):
        # The v0.2 trap: best_val_iou in metrics.json is the IoU at the
        # best-*loss* epoch, which is not the peak.
        history = _history([(5, 0.10, 0.60), (10, 0.30, 0.80), (15, 0.20, 0.70)])
        assert peak_from_history(history) == (0.80, 10)

    def test_no_validated_epochs(self):
        assert peak_from_history([{"epoch": 1, "train": {}, "val": None}]) == (0.0, None)

    def test_empty_history(self):
        assert peak_from_history([]) == (0.0, None)


class TestSummariseRun:
    def test_missing_metrics_returns_none(self, tmp_path):
        (tmp_path / "empty_run").mkdir()
        assert summarise_run(tmp_path / "empty_run") is None

    def test_reports_final_and_oracle_peak_separately(self, tmp_path):
        run = _write_run(
            tmp_path, "r", _history([(5, 0.1, 0.60), (10, 0.3, 0.80)]), final_iou=0.55
        )
        summary = summarise_run(run)
        assert summary["final_iou"] == 0.55
        assert summary["peak_iou_oracle"] == 0.80
        assert summary["peak_epoch_oracle"] == 10

    def test_works_without_per_sample_stats(self, tmp_path):
        run = _write_run(tmp_path, "r", _history([(5, 0.1, 0.6)]), final_iou=0.6)
        assert "eval_samples" not in summarise_run(run)

    def test_per_sample_stats_add_distribution_columns(self, tmp_path):
        run = _write_run(tmp_path, "r", _history([(5, 0.1, 0.6)]), final_iou=0.6)
        _write_stats(
            run,
            [_row("a", 10, 10, 1.0), _row("b", 10, 0, 0.0), _row("c", 10, 8, 0.6)],
            epoch=5,
        )
        s = summarise_run(run)
        assert s["eval_samples"] == 3
        assert s["zero_iou_count"] == 1
        assert s["frac_iou_ge_050"] == round(2 / 3, 4)
        assert s["median_iou"] == 0.6

    def test_source_pixel_columns_appear_only_when_every_row_has_them(self, tmp_path):
        run = _write_run(tmp_path, "r", _history([(5, 0.1, 0.6)]), final_iou=0.6)
        _write_stats(
            run,
            [
                _row("a", 10, 10, 1.0, abs_error_source_px=8.0, gt_area_source_px=40.0),
                _row("b", 10, 10, 1.0),  # missing the source fields
            ],
            epoch=5,
        )
        assert "mean_abs_error_source_px" not in summarise_run(run)

    def test_source_pixel_columns_are_averaged(self, tmp_path):
        run = _write_run(tmp_path, "r", _history([(5, 0.1, 0.6)]), final_iou=0.6)
        _write_stats(
            run,
            [
                _row("a", 10, 10, 1.0, abs_error_source_px=8.0, gt_area_source_px=40.0),
                _row("b", 10, 10, 1.0, abs_error_source_px=4.0, gt_area_source_px=60.0),
            ],
            epoch=5,
        )
        s = summarise_run(run)
        assert s["mean_abs_error_source_px"] == 6.0
        assert s["mean_gt_area_source_px"] == 50.0

    def test_fold_configuration_is_carried_through(self, tmp_path):
        run = _write_run(
            tmp_path,
            "r",
            _history([(5, 0.1, 0.6)]),
            final_iou=0.6,
            config={
                "dataset": {
                    "train_sheets": ["A", "B"],
                    "eval_sheets": ["C"],
                    "train_subsample": 51,
                    "window_px": 512,
                }
            },
        )
        s = summarise_run(run)
        assert s["eval_sheets"] == "C"
        assert s["train_sheets"] == "A,B"
        assert s["train_subsample"] == 51
        assert s["window_px"] == 512

    def test_latest_epoch_stats_are_used(self, tmp_path):
        run = _write_run(tmp_path, "r", _history([(5, 0.1, 0.6)]), final_iou=0.6)
        _write_stats(run, [_row("a", 10, 10, 1.0)], epoch=5)
        _write_stats(run, [_row("a", 10, 10, 0.0), _row("b", 10, 0, 0.0)], epoch=50)
        assert summarise_run(run)["eval_samples"] == 2


class TestWriteCsv:
    def test_unions_columns_across_rows(self, tmp_path):
        out = tmp_path / "out.csv"
        write_csv(out, [{"run": "a", "x": 1}, {"run": "b", "y": 2}])
        rows = list(csv.DictReader(out.open(encoding="utf-8")))
        assert set(rows[0]) == {"run", "x", "y"}
        assert rows[0]["y"] == ""  # filled, not dropped

    def test_no_rows_writes_nothing(self, tmp_path):
        out = tmp_path / "out.csv"
        write_csv(out, [])
        assert not out.exists()
