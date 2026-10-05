"""Один финальный test прогон уже обученного checkpoint без повторного training."""

import argparse
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader

from .benchmark import measure_inference
from .data import LandCoverTileDataset
from .models import create_model
from .postprocess import export_run_artifacts
from .run_utils import load_config, save_json
from .train import evaluate
from .transforms import validation_transform


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--data-root", required=True, type=Path)
    return parser.parse_args()


def main():
    args = parse_args()
    run_dir = args.run_dir
    if (run_dir / "test_metrics.csv").exists():
        raise FileExistsError("Test metrics already exist: final test must not be rerun.")

    config = load_config(run_dir / "config.yaml")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    image_dir, mask_dir = args.data_root / "images", args.data_root / "masks"
    tile_index = pd.read_csv(run_dir / "tile_index.csv")
    test_index = tile_index.query("split == 'test'").reset_index(drop=True)
    test_dataset = LandCoverTileDataset(test_index, image_dir, mask_dir, config["tile_size"], validation_transform())

    loader_kwargs = {
        "batch_size": config["batch_size"],
        "num_workers": config["num_workers"],
        "pin_memory": True,
        "persistent_workers": config["num_workers"] > 0,
    }
    if config["num_workers"] > 0:
        loader_kwargs["prefetch_factor"] = 2
    test_loader = DataLoader(test_dataset, shuffle=False, **loader_kwargs)

    model = create_model(
        config["architecture"], config["encoder_name"], config["encoder_weights"], config["num_classes"]
    ).to(device)
    checkpoint = torch.load(run_dir / "best_checkpoint.pth", map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])

    scores, per_scene = evaluate(model, test_loader, device, config["num_classes"], config["ignore_index"])
    pd.DataFrame([scores]).to_csv(run_dir / "test_metrics.csv", index=False)
    pd.DataFrame(per_scene).to_csv(run_dir / "test_per_scene_metrics.csv", index=False)
    efficiency = measure_inference(model, device, config["tile_size"])
    save_json(efficiency, run_dir / "efficiency.json")
    exported = export_run_artifacts(model, test_dataset, device, run_dir)

    print(pd.Series(scores).sort_index())
    print("Efficiency:", efficiency)
    print("Export:", exported)


if __name__ == "__main__":
    main()
