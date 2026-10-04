"""Единообразный замер вычислительной эффективности моделей на одном GPU."""

from time import perf_counter

import torch


@torch.no_grad()
def measure_inference(model, device, tile_size=512, warmup_runs=20, timed_runs=100):
    """Batch=1 latency и peak VRAM после warm-up.

    Замер корректно сопоставим только между моделями на одном GPU, с тем же
    размером входа, числом прогревочных и измеряемых запусков.
    """
    model.eval()
    sample = torch.zeros((1, 3, tile_size, tile_size), device=device)
    is_cuda = device.type == "cuda"

    for _ in range(warmup_runs):
        model(sample)
    if is_cuda:
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)

    started_at = perf_counter()
    for _ in range(timed_runs):
        model(sample)
    if is_cuda:
        torch.cuda.synchronize(device)

    elapsed_seconds = perf_counter() - started_at
    parameters = sum(parameter.numel() for parameter in model.parameters())
    result = {
        "parameters": parameters,
        "parameters_m": parameters / 1e6,
        "latency_ms_batch1": 1000 * elapsed_seconds / timed_runs,
        "warmup_runs": warmup_runs,
        "timed_runs": timed_runs,
        "tile_size": tile_size,
        "device": str(device),
    }
    if is_cuda:
        result["peak_vram_mb"] = torch.cuda.max_memory_allocated(device) / 1024**2
    return result
