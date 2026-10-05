"""CLI runner одного контролируемого эксперимента в Kaggle."""

import argparse
import json
from pathlib import Path
import subprocess

import pandas as pd
import torch
from torch.utils.data import DataLoader

from .data import LandCoverTileDataset
from .index import build_tile_index
from .models import create_model
from .run_utils import create_run_dir, load_config, save_config, save_json, set_seed
from .train import evaluate, train_one_epoch
from .transforms import training_transform, validation_transform


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--runs-root", default=Path("/kaggle/working/runs"), type=Path)
    parser.add_argument("--test", action="store_true", help="run the locked final test evaluation")
    return parser.parse_args()


def main():
    args = parse_args()
    config = load_config(args.config)
    project_dir = args.config.parent.parent
    set_seed(config["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with open(project_dir / "configs" / config["split_file"], encoding="utf-8") as handle:
        scene_split = json.load(handle)

    image_dir, mask_dir = args.data_root / "images", args.data_root / "masks"
    tile_index = build_tile_index(image_dir, scene_split, config["tile_size"])
    run_dir = create_run_dir(args.runs_root, config["experiment_name"])
    tile_index.to_csv(run_dir / "tile_index.csv", index=False)
    save_config(config, run_dir)

    try:
        revision = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=project_dir, text=True
        ).strip()
    except subprocess.CalledProcessError:
        revision = "unknown"
    save_json({"device": str(device), "code_revision": revision}, run_dir / "environment.json")

    train_index = tile_index.query("split == 'train'").reset_index(drop=True)
    val_index = tile_index.query("split == 'val'").reset_index(drop=True)
    test_index = tile_index.query("split == 'test'").reset_index(drop=True)
    train_dataset = LandCoverTileDataset(train_index, image_dir, mask_dir, config["tile_size"], training_transform())
    val_dataset = LandCoverTileDataset(val_index, image_dir, mask_dir, config["tile_size"], validation_transform())
    test_dataset = LandCoverTileDataset(test_index, image_dir, mask_dir, config["tile_size"], validation_transform())

    loader_kwargs = {
        "batch_size": config["batch_size"],
        "num_workers": config["num_workers"],
        "pin_memory": True,
        "persistent_workers": config["num_workers"] > 0,
    }
    if config["num_workers"] > 0:
        loader_kwargs["prefetch_factor"] = 2
    train_loader = DataLoader(train_dataset, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_dataset, shuffle=False, **loader_kwargs)
    test_loader = DataLoader(test_dataset, shuffle=False, **loader_kwargs)

    model = create_model(
        config["architecture"], config["encoder_name"], config["encoder_weights"], config["num_classes"]
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"]
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", patience=3, factor=0.5)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    save_json({"parameters": parameter_count, "parameters_m": parameter_count / 1e6}, run_dir / "model.json")

    print(f"Run: {run_dir}")
    print(f"Device: {device}; train={len(train_dataset)}, val={len(val_dataset)}, test={len(test_dataset)}")
    print(f"Model: {config['architecture']} + {config['encoder_name']} | {parameter_count / 1e6:.2f} M parameters")

    best_val_miou, no_improvement, history = float("-inf"), 0, []
    for epoch in range(1, config["max_epochs"] + 1):
        train_scores = train_one_epoch(model, train_loader, optimizer, device, config["num_classes"], config["ignore_index"])
        val_scores, _ = evaluate(model, val_loader, device, config["num_classes"], config["ignore_index"])
        scheduler.step(val_scores["foreground_miou"])
        row = {"epoch": epoch, "learning_rate": optimizer.param_groups[0]["lr"]}
        row.update({f"train_{key}": value for key, value in train_scores.items()})
        row.update({f"val_{key}": value for key, value in val_scores.items()})
        history.append(row)
        pd.DataFrame(history).to_csv(run_dir / "history.csv", index=False)

        if val_scores["foreground_miou"] > best_val_miou:
            best_val_miou, no_improvement = val_scores["foreground_miou"], 0
            torch.save(
                {"epoch": epoch, "model_state_dict": model.state_dict(), "config": config},
                run_dir / "best_checkpoint.pth",
            )
        else:
            no_improvement += 1
        print(
            f"Epoch {epoch:02d} | train loss={train_scores['loss']:.4f} | "
            f"val fg-mIoU={val_scores['foreground_miou']:.4f} | best={best_val_miou:.4f}"
        )
        if no_improvement >= config["early_stopping_patience"]:
            print("Early stopping: validation foreground mIoU has not improved.")
            break

    save_json({"best_val_foreground_miou": best_val_miou, "epochs_completed": len(history)}, run_dir / "summary.json")
    if not args.test:
        print("Training completed. Test evaluation was intentionally not run.")
        return

    checkpoint = torch.load(run_dir / "best_checkpoint.pth", map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    test_scores, per_scene = evaluate(model, test_loader, device, config["num_classes"], config["ignore_index"])
    pd.DataFrame([test_scores]).to_csv(run_dir / "test_metrics.csv", index=False)
    pd.DataFrame(per_scene).to_csv(run_dir / "test_per_scene_metrics.csv", index=False)
    print(pd.Series(test_scores).sort_index())


if __name__ == "__main__":
    main()
