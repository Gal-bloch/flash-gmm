"""Memory-bounded Flash-GMM training example for one billion SIFT vectors.

The input is a standard ``u8bin`` file: two little-endian uint32 header
values (row count and dimension), followed by row-major uint8 vectors.
Chunks are staged through pinned host memory and accumulated in FP64. Model
parameters remain fixed while all chunks of an EM iteration are processed.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch

from flash_gmm import flash_gmm_estep, flash_gmm_mstep


class DoubleBufferedU8Streamer:
    """Overlap RAM copies, host-to-device transfer, and uint8-to-FP32 cast."""

    def __init__(self, data: np.ndarray, chunk_rows: int, device: torch.device):
        self.data = data
        self.chunk_rows = chunk_rows
        self.device = device
        dimension = data.shape[1]
        self.host = [
            torch.empty((chunk_rows, dimension), dtype=torch.uint8, pin_memory=True)
            for _ in range(2)
        ]
        self.device_u8 = [
            torch.empty((chunk_rows, dimension), dtype=torch.uint8, device=device)
            for _ in range(2)
        ]
        self.device_fp32 = [
            torch.empty((chunk_rows, dimension), dtype=torch.float32, device=device)
            for _ in range(2)
        ]
        self.copy_stream = torch.cuda.Stream()
        self.ready = [torch.cuda.Event(), torch.cuda.Event()]
        self.done = [None, None]

    def stage(self, start: int, slot: int) -> int:
        end = min(start + self.chunk_rows, len(self.data))
        rows = end - start
        if self.done[slot] is not None:
            self.done[slot].synchronize()
        np.copyto(self.host[slot][:rows].numpy(), self.data[start:end])
        with torch.cuda.stream(self.copy_stream):
            self.device_u8[slot][:rows].copy_(
                self.host[slot][:rows], non_blocking=True
            )
            self.device_fp32[slot][:rows].copy_(self.device_u8[slot][:rows])
            self.ready[slot].record(self.copy_stream)
        return rows


def open_u8bin(path: Path, limit: int | None) -> np.ndarray:
    with path.open("rb") as handle:
        rows, dimension = np.fromfile(handle, dtype="<u4", count=2)
    rows, dimension = int(rows), int(dimension)
    expected_bytes = 8 + rows * dimension
    if path.stat().st_size != expected_bytes:
        raise ValueError(
            f"u8bin size mismatch: got {path.stat().st_size}, "
            f"expected {expected_bytes} bytes"
        )
    selected_rows = rows if limit is None else min(limit, rows)
    return np.memmap(
        path,
        dtype=np.uint8,
        mode="r",
        offset=8,
        shape=(rows, dimension),
    )[:selected_rows]


def load_centers(path: Path, key: str, k: int, dimension: int) -> np.ndarray:
    artifact = np.load(path)
    if isinstance(artifact, np.lib.npyio.NpzFile):
        try:
            centers = np.asarray(artifact[key], dtype=np.float32)
        finally:
            artifact.close()
    else:
        centers = np.asarray(artifact, dtype=np.float32)
    expected = (k, dimension)
    if centers.shape != expected:
        raise ValueError(f"center shape is {centers.shape}; expected {expected}")
    return centers


def nearest_centers(
    vectors: torch.Tensor, centers: torch.Tensor, center_block: int
) -> torch.Tensor:
    vector_norm = vectors.square().sum(1)
    best_distance = torch.full(
        (len(vectors),), float("inf"), device=vectors.device
    )
    best_index = torch.zeros(
        (len(vectors),), dtype=torch.int64, device=vectors.device
    )
    for start in range(0, len(centers), center_block):
        center_chunk = centers[start : start + center_block]
        distance = (
            vector_norm[:, None]
            - 2.0 * (vectors @ center_chunk.T)
            + center_chunk.square().sum(1)[None, :]
        )
        block_distance, block_index = distance.min(1)
        improve = block_distance < best_distance
        best_distance = torch.where(improve, block_distance, best_distance)
        best_index = torch.where(improve, block_index + start, best_index)
    return best_index


def hard_initialize(
    data: np.ndarray,
    centers: torch.Tensor,
    chunk_rows: int,
    center_block: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Initialize weights and spherical variances from fixed centroids."""
    device = centers.device
    rows, dimension = data.shape
    k = len(centers)
    stream = DoubleBufferedU8Streamer(data, chunk_rows, device)
    counts = torch.zeros(k, dtype=torch.float64, device=device)
    sums = torch.zeros((k, dimension), dtype=torch.float64, device=device)
    second_moment = torch.zeros(k, dtype=torch.float64, device=device)

    current_rows = stream.stage(0, 0)
    chunk_count = (rows + chunk_rows - 1) // chunk_rows
    for chunk_index in range(chunk_count):
        slot = chunk_index & 1
        start = chunk_index * chunk_rows
        torch.cuda.current_stream().wait_event(stream.ready[slot])
        vectors = stream.device_fp32[slot][:current_rows]
        assignment = nearest_centers(vectors, centers, center_block)

        chunk_sums = torch.zeros(
            (k, dimension), dtype=torch.float32, device=device
        )
        chunk_second = torch.zeros(k, dtype=torch.float32, device=device)
        chunk_sums.index_add_(0, assignment, vectors)
        chunk_second.index_add_(0, assignment, vectors.square().sum(1))
        sums.add_(chunk_sums.double())
        second_moment.add_(chunk_second.double())
        counts.add_(torch.bincount(assignment, minlength=k).double())

        done = torch.cuda.Event()
        done.record()
        stream.done[slot] = done
        if chunk_index + 1 < chunk_count:
            next_slot = (chunk_index + 1) & 1
            current_rows = stream.stage(start + current_rows, next_slot)

    torch.cuda.synchronize()
    safe_counts = counts.clamp_min(1e-10)
    means = sums / safe_counts[:, None]
    scatter = (second_moment - counts * means.square().sum(1)).clamp_min(0)
    variance = (scatter / (dimension * safe_counts)).clamp_min(1e-6)
    log_weights = (counts / counts.sum()).clamp_min(1e-10).log()
    empty = counts < 1
    if empty.any():
        means[empty] = centers[empty].double()
        variance[empty] = variance[~empty].mean()
        log_weights[empty] = math.log(1.0 / k)
    return means.float(), variance.log().float(), log_weights.float()


def fit_chunked(
    data: np.ndarray,
    parameters: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    iterations: int,
    chunk_rows: int,
) -> tuple[tuple[torch.Tensor, torch.Tensor, torch.Tensor], dict]:
    """Run EM without materializing either X or the N-by-K matrix on GPU."""
    means, log_variance, log_weights = parameters
    device = means.device
    rows, dimension = data.shape
    k = len(means)
    stream = DoubleBufferedU8Streamer(data, chunk_rows, device)
    iteration_seconds = []
    average_log_likelihood = []
    allocated_before = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()

    # Compile and warm the exact full-chunk kernel shape outside timing.
    warm_rows = stream.stage(0, 0)
    torch.cuda.current_stream().wait_event(stream.ready[0])
    flash_gmm_estep(
        stream.device_fp32[0][:warm_rows], means, log_variance, log_weights
    )
    torch.cuda.synchronize()

    for iteration in range(iterations):
        started = time.perf_counter()
        count_total = torch.zeros(k, dtype=torch.float64, device=device)
        first_total = torch.zeros(
            (k, dimension), dtype=torch.float64, device=device
        )
        scatter_total = torch.zeros(k, dtype=torch.float64, device=device)
        log_likelihood_total = torch.zeros((), dtype=torch.float64, device=device)

        current_rows = stream.stage(0, 0)
        chunk_count = (rows + chunk_rows - 1) // chunk_rows
        for chunk_index in range(chunk_count):
            slot = chunk_index & 1
            start = chunk_index * chunk_rows
            torch.cuda.current_stream().wait_event(stream.ready[slot])
            vectors = stream.device_fp32[slot][:current_rows]
            log_z, counts, first, scatter = flash_gmm_estep(
                vectors, means, log_variance, log_weights
            )
            count_total.add_(counts.double())
            first_total.add_(first.double())
            scatter_total.add_(scatter.double())
            log_likelihood_total.add_(log_z.double().sum())

            done = torch.cuda.Event()
            done.record()
            stream.done[slot] = done
            if chunk_index + 1 < chunk_count:
                next_slot = (chunk_index + 1) & 1
                current_rows = stream.stage(start + current_rows, next_slot)

        torch.cuda.synchronize()
        means, log_variance, log_weights = flash_gmm_mstep(
            count_total, first_total, scatter_total, means.double(), dimension
        )
        means = means.float()
        log_variance = log_variance.float()
        log_weights = log_weights.float()
        torch.cuda.synchronize()

        elapsed = time.perf_counter() - started
        avg_ll = float(log_likelihood_total / rows)
        iteration_seconds.append(elapsed)
        average_log_likelihood.append(avg_ll)
        print(
            f"iteration={iteration + 1}/{iterations} "
            f"seconds={elapsed:.3f} avg_log_likelihood={avg_ll:.8f}",
            flush=True,
        )

    metrics = {
        "rows": rows,
        "dimension": dimension,
        "components": k,
        "iterations": iterations,
        "chunk_rows": chunk_rows,
        "iteration_seconds": iteration_seconds,
        "average_log_likelihood": average_log_likelihood,
        "incremental_peak_mib": (
            torch.cuda.max_memory_allocated() - allocated_before
        ) / 2**20,
        "gpu": torch.cuda.get_device_name(device),
    }
    return (means, log_variance, log_weights), metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--init-centers", type=Path, required=True)
    parser.add_argument("--init-centers-key", default="centroids_10")
    parser.add_argument("--rows", type=int, default=1_000_000_000)
    parser.add_argument("--components", type=int, default=10_000)
    parser.add_argument("--iterations", type=int, default=30)
    parser.add_argument("--chunk-rows", type=int, default=10_000_000)
    parser.add_argument("--center-block", type=int, default=128)
    parser.add_argument("--output", type=Path, default=Path("sift1b_run.json"))
    parser.add_argument("--save-parameters", type=Path)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("A CUDA GPU is required")

    data = open_u8bin(args.data, args.rows)
    centers_np = load_centers(
        args.init_centers, args.init_centers_key, args.components, data.shape[1]
    )
    centers = torch.from_numpy(centers_np).cuda()
    parameters = hard_initialize(
        data, centers, args.chunk_rows, args.center_block
    )
    parameters, metrics = fit_chunked(
        data, parameters, args.iterations, args.chunk_rows
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(metrics, indent=2) + "\n")
    if args.save_parameters is not None:
        args.save_parameters.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "mu": parameters[0].cpu(),
                "log_sigma_sq": parameters[1].cpu(),
                "log_pi": parameters[2].cpu(),
                **{key: metrics[key] for key in (
                    "rows", "dimension", "components", "iterations", "chunk_rows"
                )},
            },
            args.save_parameters,
        )
    print("RESULT " + json.dumps(metrics), flush=True)


if __name__ == "__main__":
    main()
