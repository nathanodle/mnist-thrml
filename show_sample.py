#!/usr/bin/env python
"""Preview a binarized MNIST sample and (optionally) score it with a trained Ising EBM."""

import argparse
from pathlib import Path

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from train_mnist_thrml import (
    free_energy_visible,
    labels_to_spin_matrix,
    load_mnist,
)


def load_checkpoint(path: Path, n_vis: int) -> tuple[tuple[jnp.ndarray, jnp.ndarray], jnp.ndarray, int]:
    """Load weights, biases, beta from an .npz checkpoint."""
    data = np.load(path)
    weights = jnp.array(data["weights"])
    biases = jnp.array(data["biases"])
    beta = jnp.array(data["beta"] if "beta" in data else 1.0, dtype=jnp.float32)
    n_hidden = weights.size // n_vis
    if weights.size % n_vis != 0:
        raise ValueError("weights array size is not divisible by n_vis; checkpoint corrupted?")
    return (weights, biases), beta, n_hidden


def main() -> None:
    parser = argparse.ArgumentParser(description="Display an MNIST digit and score it with the trained Ising EBM.")
    parser.add_argument("--split", choices=["train", "test"], default="test", help="Dataset split to draw from.")
    parser.add_argument("--index", type=int, default=0, help="Index of the sample to preview.")
    parser.add_argument("--binarize-thresh", type=float, default=0.5, help="Threshold for converting pixels to spins.")
    parser.add_argument("--data-dir", type=str, default=None, help="Optional TFDS data directory override.")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to .npz file containing weights/biases/beta saved after training (see README notes).",
    )
    parser.add_argument("--n-classes", type=int, default=10, help="Number of label spins used during training.")
    parser.add_argument(
        "--use-label-spins",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Set to --no-use-label-spins if the checkpoint was trained without appended label spins.",
    )
    parser.add_argument(
        "--save-image",
        type=str,
        default=None,
        help="Optional destination PNG path for the preview instead of showing a window.",
    )
    args = parser.parse_args()

    (x_train, y_train), (x_test, y_test) = load_mnist(args.data_dir, args.binarize_thresh)
    xs, ys = (x_train, y_train) if args.split == "train" else (x_test, y_test)

    if args.index < 0 or args.index >= xs.shape[0]:
        raise ValueError(f"index {args.index} is outside the range 0..{xs.shape[0]-1}")

    sample = xs[args.index]
    label = int(ys[args.index])
    image = (sample.reshape(28, 28) + 1.0) / 2.0

    fig, ax = plt.subplots(figsize=(3, 3))
    ax.imshow(image, cmap="gray")
    ax.set_title(f"{args.split} idx={args.index}, label={label}")
    ax.axis("off")

    energy_info = "no checkpoint provided"
    if args.checkpoint is not None:
        n_pixels = sample.shape[0]
        n_vis = n_pixels + (args.n_classes if args.use_label_spins else 0)
        params, beta, n_hidden = load_checkpoint(Path(args.checkpoint), n_vis=n_vis)

        if args.use_label_spins:
            label_spin = labels_to_spin_matrix(np.array([label], dtype=np.int32), args.n_classes)[0]
            vis_sample = np.concatenate([sample, label_spin])
        else:
            vis_sample = sample

        vis = jnp.array(vis_sample[None, :])
        fe = float(jnp.asarray(free_energy_visible(params, beta, vis, n_vis, n_hidden)).squeeze())
        energy_info = f"free-energy surrogate = {fe:.4f}"
        print(f"{args.split} sample {args.index} label={label} -> {energy_info}")
    else:
        print(f"{args.split} sample {args.index} label={label} -> {energy_info}")

    fig.text(0.5, 0.02, energy_info, ha="center", va="bottom", fontsize=9)

    if args.save_image:
        output_path = Path(args.save_image)
        fig.savefig(output_path, bbox_inches="tight")
        print(f"Saved preview to {output_path}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
