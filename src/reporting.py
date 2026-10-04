"""Сбор сопоставимых итогов из папок завершённых experiment runs."""

from pathlib import Path

import pandas as pd


def load_run_summary(run_dir):
    run_dir = Path(run_dir)
    config = pd.read_csv(run_dir / "history.csv")
    test = pd.read_csv(run_dir / "test_metrics.csv").iloc[0].to_dict()
    best_epoch = int(config["val_foreground_miou"].idxmax()) + 1
    best_row = config.iloc[best_epoch - 1].to_dict()
    return {
        "run_dir": run_dir.name,
        "best_epoch": best_epoch,
        **{f"test_{key}": value for key, value in test.items()},
        "best_val_foreground_miou": best_row["val_foreground_miou"],
    }


def comparison_table(run_dirs):
    """Одна строка на модель для финальной таблицы отчёта."""
    table = pd.DataFrame([load_run_summary(run_dir) for run_dir in run_dirs])
    sort_column = "test_foreground_miou"
    return table.sort_values(sort_column, ascending=False).reset_index(drop=True)
