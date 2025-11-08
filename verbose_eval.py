#!/usr/bin/env python
"""Verbose accuracy probe for THRML MNIST checkpoints."""

import argparse
from pathlib import Path

import jax.numpy as jnp
import numpy as np

from train_mnist_thrml import (
    load_mnist,
    load_params_npz,
    predict_labels,
)


def select_indices(total: int, count: int, offset: int, random_sample: bool, seed: int) -> np.ndarray:
    if count <= 0:
        raise ValueError("sample-count must be positive")
    count = min(count, total)
    if random_sample:
        rng = np.random.default_rng(seed)
        return np.sort(rng.choice(total, size=count, replace=False))
    start = np.clip(offset, 0, max(total - count, 0))
    return np.arange(start, start + count)


def main() -> None:
    parser = argparse.ArgumentParser(description="Print per-sample predictions for a THRML MNIST checkpoint.")
    parser.add_argument("--checkpoint", required=True, help="Path to .npz checkpoint file.")
    parser.add_argument("--split", choices=["train", "test"], default="test")
    parser.add_argument("--sample-count", type=int, default=20, help="Number of samples to inspect.")
    parser.add_argument("--offset", type=int, default=0, help="Starting index when not sampling randomly.")
    parser.add_argument("--random", action=argparse.BooleanOptionalAction, default=False, help="Randomly sample indices instead of a contiguous block.")
    parser.add_argument("--seed", type=int, default=0, help="RNG seed when --random is enabled.")
    parser.add_argument("--binarize-thresh", type=float, default=0.5)
    parser.add_argument("--data-dir", type=str, default=None)
    parser.add_argument("--n-classes", type=int, default=10)
    parser.add_argument("--use-label-spins", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    weights, biases, beta = load_params_npz(Path(args.checkpoint))

    (x_train_pix, y_train), (x_test_pix, y_test) = load_mnist(args.data_dir, args.binarize_thresh)
    xs_pix, ys = (x_train_pix, y_train) if args.split == "train" else (x_test_pix, y_test)

    n_pixels = xs_pix.shape[1]
    n_vis_expected = n_pixels + (args.n_classes if args.use_label_spins else 0)
    if weights.size % n_vis_expected != 0:
        raise ValueError(
            "Checkpoint weight matrix is incompatible with the requested label-spin configuration."
        )
    n_hidden = weights.size // n_vis_expected

    idx = select_indices(xs_pix.shape[0], args.sample_count, args.offset, args.random, args.seed)
    batch_pixels = jnp.array(xs_pix[idx])
    preds = np.array(predict_labels((weights, biases), beta, batch_pixels, n_pixels, n_hidden, args.n_classes))
    truths = ys[idx]
    correct_mask = preds == truths
    accuracy = float(np.mean(correct_mask))

    print(f"checkpoint: {args.checkpoint}")
    print(f"split: {args.split}, samples: {len(idx)}, inferred n_hidden: {n_hidden}, label spins: {args.use_label_spins}")
    print(f"batch accuracy: {accuracy * 100:.2f}%\n")
    print(f"{'idx':>6}  {'pred':>4}  {'truth':>5}  {'correct':>7}")
    for sample_idx, pred, truth, ok in zip(idx, preds, truths, correct_mask):
        print(f"{sample_idx:6d}  {pred:4d}  {truth:5d}  {str(bool(ok)):>7}")


if __name__ == "__main__":
    main()
