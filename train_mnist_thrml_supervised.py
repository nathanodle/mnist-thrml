#!/usr/bin/env python
"""Train an Ising-style EBM on MNIST using THRML with clamped (conditional) training.

- Positive phase: clamp pixels (always) and, if label spins are enabled, clamp labels; sample hidden.
- Negative phase: clamp pixels; sample labels (if any) and hidden.

When --use-label-spins=false, this reduces to the original joint training where the
visible block is clamped in the positive phase and both vis+hid are sampled in the
negative phase.
"""

import argparse
import os
import time
from functools import partial
from pathlib import Path
from typing import Iterable, Tuple

from env_helpers import ensure_cuda_shared_libs_visible

os.environ.setdefault("JAX_PLATFORMS", "cuda,cpu")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
ensure_cuda_shared_libs_visible()

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
from jax import random
from tqdm import trange

from thrml.block_management import Block
from thrml.block_sampling import SamplingSchedule
from thrml.models.ising import IsingEBM, IsingTrainingSpec, estimate_kl_grad, estimate_moments, hinton_init
from thrml.pgm import SpinNode

Array = jnp.ndarray

# Compatibility for older THRML versions expecting jax.tree.flatten_with_path
if not hasattr(jax.tree, "flatten_with_path"):
    jax.tree.flatten_with_path = jax.tree_util.tree_flatten_with_path  # type: ignore[attr-defined]


def load_mnist(
    data_dir: str | None, threshold: float
) -> Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]:
    """Load MNIST splits from TFDS, normalize to [0, 1], then map to spins in {-1, +1}."""
    builder_kwargs = dict(as_supervised=True, batch_size=-1)
    if data_dir is not None:
        builder_kwargs["data_dir"] = data_dir

    train_ds = tfds.load("mnist", split="train", **builder_kwargs)
    test_ds = tfds.load("mnist", split="test", **builder_kwargs)

    x_train, y_train = tfds.as_numpy(train_ds)
    x_test, y_test = tfds.as_numpy(test_ds)

    def _prep(images: np.ndarray) -> np.ndarray:
        images = images.astype(np.float32) / 255.0
        images = (images > threshold).astype(np.float32)
        images = images.reshape(images.shape[0], -1)
        return images * 2.0 - 1.0

    return (_prep(x_train), y_train.astype(np.int32)), (_prep(x_test), y_test.astype(np.int32))


def labels_to_spin_matrix(labels: np.ndarray, n_classes: int) -> np.ndarray:
    """Map integer labels to {-1, +1} one-hot spin vectors."""
    eye = np.eye(n_classes, dtype=np.float32)
    return eye[labels] * 2.0 - 1.0


def append_label_spins(features: np.ndarray, labels: np.ndarray, n_classes: int) -> np.ndarray:
    """Concatenate label spin vectors to the visible pixel spins."""
    label_spins = labels_to_spin_matrix(labels, n_classes)
    return np.concatenate([features, label_spins], axis=1)


def make_batches(
    rng: np.random.Generator, xs: np.ndarray, ys: np.ndarray, batch_size: int, drop_last: bool = True
) -> Iterable[Tuple[np.ndarray, np.ndarray]]:
    """Yield shuffled mini-batches with optional remainder drop to keep shapes static."""
    idx = rng.permutation(xs.shape[0])
    if drop_last:
        limit = (xs.shape[0] // batch_size) * batch_size
        idx = idx[:limit]
    for start in range(0, idx.shape[0], batch_size):
        sl = idx[start : start + batch_size]
        if sl.size < batch_size:
            break
        yield xs[sl], ys[sl]


def build_model(
    n_vis: int,
    n_hid: int,
    beta: float,
    *,
    use_label_spins: bool,
    n_pixels: int,
    n_classes: int,
):
    """Build Ising model and return useful blocks for clamped training.

    Returns:
        model, nodes, edges,
        block_vis_all, block_pixels, block_labels, block_hid
    """
    vis_nodes = [SpinNode() for _ in range(n_vis)]
    hid_nodes = [SpinNode() for _ in range(n_hid)]
    nodes = vis_nodes + hid_nodes
    edges = [(v, h) for v in vis_nodes for h in hid_nodes]

    dtype = jnp.float32
    biases = jnp.zeros(len(nodes), dtype=dtype)
    weights = jnp.zeros(len(edges), dtype=dtype)
    beta_arr = jnp.array(beta, dtype=dtype)

    model = IsingEBM(nodes, edges, biases, weights, beta_arr)

    # Blocks
    block_vis_all = Block(vis_nodes)
    block_hid = Block(hid_nodes)

    if use_label_spins:
        pixels_nodes = vis_nodes[:n_pixels]
        labels_nodes = vis_nodes[n_pixels : n_pixels + n_classes]
        block_pixels = Block(pixels_nodes)
        block_labels = Block(labels_nodes)
    else:
        # No label spins present; treat all visibles as "pixels"
        block_pixels = block_vis_all
        block_labels = None

    return model, nodes, edges, block_vis_all, block_pixels, block_labels, block_hid


def make_training_spec(
    model: IsingEBM,
    *,
    block_pixels: Block,
    block_labels: Block | None,
    block_hid: Block,
    schedule_pos: SamplingSchedule,
    schedule_neg: SamplingSchedule,
    use_label_spins: bool,
) -> IsingTrainingSpec:
    """Configure clamped training.

    - Positive: clamp pixels and (if enabled) labels; sample hidden.
    - Negative: clamp pixels; sample labels (if any) and hidden.

    Implementation detail:
    * Put PIXELS in conditioning_blocks  -> clamped in BOTH phases.
    * Put LABELS in data_blocks          -> clamped ONLY in the POSITIVE phase.
    """
    if use_label_spins and block_labels is not None:
        data_blocks = [block_labels]                 # labels clamped only in positive
        conditioning_blocks = [block_pixels]         # pixels clamped in both phases
        positive_sampling_blocks = [block_hid]       # hidden free (+)
        negative_sampling_blocks = [block_labels, block_hid]  # labels+hidden free (-)
    else:
        # No label spins
        data_blocks = [block_pixels]
        conditioning_blocks = []
        positive_sampling_blocks = [block_hid]
        negative_sampling_blocks = [block_pixels, block_hid]  # joint negative

    return IsingTrainingSpec(
        model,
        data_blocks=data_blocks,
        conditioning_blocks=conditioning_blocks,
        positive_sampling_blocks=positive_sampling_blocks,
        negative_sampling_blocks=negative_sampling_blocks,
        schedule_positive=schedule_pos,
        schedule_negative=schedule_neg,
    )


def estimate_kl_grad_conditioned(
    key: Array,
    training_spec: IsingTrainingSpec,
    bias_nodes: list[SpinNode],
    weight_edges: list[tuple[SpinNode, SpinNode]],
    data: list[Array],
    conditioning_values: list[Array],
    init_state_positive: list[Array],
    init_state_negative: list[Array],
) -> tuple:
    """Variant of THRML's estimator that supports per-sample conditioning data."""

    if not conditioning_values:
        raise ValueError("conditioning_values must be provided for conditioned training.")

    key_pos, key_neg = random.split(key, 2)

    clamped_pos_tree = tuple(data + conditioning_values)
    keys_pos = random.split(key_pos, init_state_positive[0].shape[:2])

    def _run_pos_chain(k_chain: Array, init_chain: list[Array]) -> tuple[Array, Array]:
        return jax.vmap(
            lambda k_sample, init_sample, clamped_sample: estimate_moments(
                k_sample,
                bias_nodes,
                weight_edges,
                training_spec.program_positive,
                training_spec.schedule_positive,
                init_sample,
                list(clamped_sample),
            )
        )(k_chain, init_chain, clamped_pos_tree)

    moms_b_pos, moms_w_pos = jax.vmap(_run_pos_chain)(keys_pos, init_state_positive)

    cond_tree = tuple(conditioning_values)
    keys_neg = random.split(key_neg, init_state_negative[0].shape[:2])

    def _run_neg_chain(k_chain: Array, init_chain: list[Array]) -> tuple[Array, Array]:
        return jax.vmap(
            lambda k_sample, init_sample, cond_sample: estimate_moments(
                k_sample,
                bias_nodes,
                weight_edges,
                training_spec.program_negative,
                training_spec.schedule_negative,
                init_sample,
                list(cond_sample),
            )
        )(k_chain, init_chain, cond_tree)

    moms_b_neg, moms_w_neg = jax.vmap(_run_neg_chain)(keys_neg, init_state_negative)

    float_type = training_spec.ebm.beta.dtype

    def _mean_except_last(x: Array) -> Array:
        axes = tuple(range(max(x.ndim - 1, 0)))
        if not axes:
            return x
        return jnp.mean(x, axis=axes, dtype=float_type)

    grad_b = -training_spec.ebm.beta * (_mean_except_last(moms_b_pos) - _mean_except_last(moms_b_neg))
    grad_w = -training_spec.ebm.beta * (_mean_except_last(moms_w_pos) - _mean_except_last(moms_w_neg))

    return grad_w, grad_b, (moms_b_pos, moms_w_pos), (moms_b_neg, moms_w_neg)


@partial(jax.jit, static_argnums=(3,))
def adam_step(params: tuple[Array, Array], grads: tuple[Array, Array], opt_state: tuple, lr: float) -> tuple:
    (mw, mb), (vw, vb), t = opt_state
    gw, gb = grads

    b1, b2, eps = 0.9, 0.999, 1e-8

    t = t + 1
    mw = b1 * mw + (1.0 - b1) * gw
    mb = b1 * mb + (1.0 - b1) * gb
    vw = b2 * vw + (1.0 - b2) * (gw * gw)
    vb = b2 * vb + (1.0 - b2) * (gb * gb)

    mw_hat = mw / (1.0 - b1**t)
    mb_hat = mb / (1.0 - b1**t)
    vw_hat = vw / (1.0 - b2**t)
    vb_hat = vb / (1.0 - b2**t)

    w, b = params
    w = w - lr * mw_hat / (jnp.sqrt(vw_hat) + eps)
    b = b - lr * mb_hat / (jnp.sqrt(vb_hat) + eps)

    new_state = ((mw, mb), (vw, vb), t)
    return (w, b), new_state


def adam_init_like(params: tuple[Array, Array]) -> tuple:
    w, b = params
    zeros_like = jnp.zeros_like
    return ((zeros_like(w), zeros_like(b)), (zeros_like(w), zeros_like(b)), jnp.array(0, dtype=w.dtype))


def free_energy_visible(params: tuple[Array, Array], beta: Array, visibles: Array, n_vis: int, n_hid: int) -> Array:
    """Compute the free energy (up to a constant) of visible spins by marginalizing hidden spins."""
    weights, biases = params
    w_matrix = weights.reshape(n_vis, n_hid)
    b_vis = biases[:n_vis]
    b_hid = biases[n_vis:]

    vis_term = jnp.sum(visibles * b_vis, axis=1)
    hidden_input = beta * (b_hid + visibles @ w_matrix)
    hidden_term = jnp.sum(jnp.logaddexp(hidden_input, -hidden_input), axis=1)
    return -beta * vis_term - hidden_term


def predict_labels(
    params: tuple[Array, Array],
    beta: Array,
    pixel_spins: Array,
    n_pixels: int,
    n_hidden: int,
    n_classes: int,
) -> Array:
    """Classify digits by scanning label spin states and picking the lowest free energy."""
    label_options = jnp.eye(n_classes, dtype=pixel_spins.dtype) * 2.0 - 1.0

    def classify_single(sample: Array) -> Array:
        sample_rep = jnp.repeat(sample[None, :], n_classes, axis=0)
        visibles = jnp.concatenate([sample_rep, label_options], axis=1)
        energies = free_energy_visible(params, beta, visibles, n_pixels + n_classes, n_hidden)
        return jnp.argmin(energies)

    return jax.vmap(classify_single)(pixel_spins)


def configure_tf_cpu() -> None:
    try:
        tf.config.set_visible_devices([], "GPU")
    except (ValueError, RuntimeError):
        pass


def save_params_npz(path: Path, params: tuple[Array, Array], beta: Array) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        path,
        weights=np.array(params[0]),
        biases=np.array(params[1]),
        beta=np.array(beta),
    )


def load_params_npz(path: Path) -> tuple[Array, Array, Array]:
    data = np.load(path)
    weights = jnp.array(data["weights"])
    biases = jnp.array(data["biases"])
    beta = jnp.array(data["beta"] if "beta" in data else 1.0, dtype=jnp.float32)
    return weights, biases, beta


def run_training(args: argparse.Namespace) -> None:
    configure_tf_cpu()

    rng = np.random.default_rng(args.seed)
    key = random.PRNGKey(args.seed)

    (x_train_pixels, y_train), (x_test_pixels, y_test) = load_mnist(args.data_dir, args.binarize_thresh)
    if args.max_train_samples is not None:
        x_train_pixels = x_train_pixels[: args.max_train_samples]
        y_train = y_train[: args.max_train_samples]
    if args.max_test_samples is not None:
        x_test_pixels = x_test_pixels[: args.max_test_samples]
        y_test = y_test[: args.max_test_samples]

    if args.use_label_spins:
        x_train = append_label_spins(x_train_pixels, y_train, args.n_classes)
        x_test = append_label_spins(x_test_pixels, y_test, args.n_classes)
    else:
        x_train = x_train_pixels
        x_test = x_test_pixels

    n_pixels = x_train_pixels.shape[1]
    n_vis = x_train.shape[1]
    checkpoint_dir = Path(args.checkpoint_dir) if args.checkpoint_dir else None

    (
        model,
        nodes,
        edges,
        _block_vis_all,
        block_pixels,
        block_labels,
        block_hid,
    ) = build_model(
        n_vis,
        args.n_hidden,
        args.beta,
        use_label_spins=args.use_label_spins,
        n_pixels=n_pixels,
        n_classes=args.n_classes,
    )

    # Positive: sample hidden only; Negative: sample labels (if any) + hidden.
    schedule = SamplingSchedule(args.n_warmup, args.n_samples, args.steps_per_sample)
    training_spec = make_training_spec(
        model,
        block_pixels=block_pixels,
        block_labels=block_labels,
        block_hid=block_hid,
        schedule_pos=schedule,
        schedule_neg=schedule,
        use_label_spins=args.use_label_spins,
    )

    params = (model.weights, model.biases)
    opt_state = adam_init_like(params)

    if args.load_params:
        load_w, load_b, load_beta = load_params_npz(Path(args.load_params))
        if load_w.shape != params[0].shape or load_b.shape != params[1].shape:
            raise ValueError(
                f"Loaded params shape mismatch: expected weights {params[0].shape}, biases {params[1].shape}; got {load_w.shape}, {load_b.shape}"
            )
        params = (load_w, load_b)
        model = eqx.tree_at(lambda m: m.weights, model, load_w)
        model = eqx.tree_at(lambda m: m.biases, model, load_b)
        model = eqx.tree_at(lambda m: m.beta, model, load_beta)
        training_spec = make_training_spec(
            model,
            block_pixels=block_pixels,
            block_labels=block_labels,
            block_hid=block_hid,
            schedule_pos=schedule,
            schedule_neg=schedule,
            use_label_spins=args.use_label_spins,
        )

    n_batches = max(1, x_train.shape[0] // args.batch_size)
    print(
        f"Training with {n_batches} batches/epoch, batch size {args.batch_size}, "
        f"n_hidden {args.n_hidden}, label_spins={args.use_label_spins}"
    )

    for epoch in range(1, args.epochs + 1):
        epoch_losses = []
        iterator = make_batches(rng, x_train, y_train, args.batch_size, drop_last=True)
        pbar = trange(n_batches, desc=f"epoch {epoch}", leave=False)
        for step_idx, (xb, _yb) in enumerate(iterator):
            xb = jnp.array(xb)

            if args.use_label_spins:
                # Slice out pixels and label spins from concatenated visibles
                xb_pixels = xb[:, :n_pixels]
                xb_labels = xb[:, n_pixels : n_pixels + args.n_classes]

                # Boolean clamps
                pixels_bool = jnp.array(xb_pixels > 0, dtype=jnp.bool_)
                labels_bool = jnp.array(xb_labels > 0, dtype=jnp.bool_)

                pos_shape = (args.n_pos_chains, xb.shape[0])  # (chains, batch)
                neg_shape = (args.n_neg_chains, xb.shape[0])  # (chains, batch)

                data_list = [labels_bool]
                cond_list = [pixels_bool]
            else:
                # No label spins: clamp all visibles in positive; joint negative
                vis_bool = jnp.array(xb > 0, dtype=jnp.bool_)
                data_list = [vis_bool]
                cond_list = []
                pos_shape = (args.n_pos_chains, xb.shape[0])
                neg_shape = (args.n_neg_chains,)

            key, k_pos_init, k_neg_init, k_grad = random.split(key, 4)

            init_pos = hinton_init(
                k_pos_init, model, training_spec.program_positive.gibbs_spec.free_blocks, pos_shape
            )
            init_neg = hinton_init(
                k_neg_init, model, training_spec.program_negative.gibbs_spec.free_blocks, neg_shape
            )

            grad_fn = estimate_kl_grad_conditioned if cond_list else estimate_kl_grad
            grad_w, grad_b, (moms_b_pos, moms_w_pos), (moms_b_neg, moms_w_neg) = grad_fn(
                k_grad,
                training_spec,
                nodes,
                edges,
                data=data_list,
                conditioning_values=cond_list,
                init_state_positive=init_pos,
                init_state_negative=init_neg,
            )

            grads = (grad_w, grad_b)
            params, opt_state = adam_step(params, grads, opt_state, lr=args.lr)
            model = eqx.tree_at(lambda m: m.weights, model, params[0])
            model = eqx.tree_at(lambda m: m.biases, model, params[1])

            # Rebuild spec with updated model tree (keeps schedules, blocks the same)
            training_spec = make_training_spec(
                model,
                block_pixels=block_pixels,
                block_labels=block_labels,
                block_hid=block_hid,
                schedule_pos=schedule,
                schedule_neg=schedule,
                use_label_spins=args.use_label_spins,
            )

            # Lightweight proxy to track progress; not the actual likelihood.
            pos_nodes_mean = jnp.mean(moms_b_pos, axis=(0, 1))
            neg_nodes_mean = jnp.mean(moms_b_neg, axis=0)
            pos_edges_mean = jnp.mean(moms_w_pos, axis=(0, 1))
            neg_edges_mean = jnp.mean(moms_w_neg, axis=0)
            loss_proxy = jnp.mean((pos_nodes_mean - neg_nodes_mean) ** 2) + jnp.mean(
                (pos_edges_mean - neg_edges_mean) ** 2
            )
            epoch_losses.append(float(loss_proxy))

            pbar.set_postfix(loss=f"{epoch_losses[-1]:.4f}")
            pbar.update(1)
        pbar.close()
        mean_loss = float(np.mean(epoch_losses))
        print(f"epoch {epoch}: mean proxy loss {mean_loss:.6f}")
        if checkpoint_dir:
            ckpt_path = checkpoint_dir / f"params_epoch{epoch:04d}.npz"
            save_params_npz(ckpt_path, params, model.beta)
            print(f"Saved checkpoint to {ckpt_path}")

    # Evaluation: approximate free energy on a held-out subset
    eval_vis_train = jnp.array(x_train[: args.eval_samples])
    eval_vis_test = jnp.array(x_test[: args.eval_samples])
    fe_train = float(jnp.mean(free_energy_visible(params, model.beta, eval_vis_train, n_vis, args.n_hidden)))
    fe_test = float(jnp.mean(free_energy_visible(params, model.beta, eval_vis_test, n_vis, args.n_hidden)))

    print("Training complete")
    print(f"free-energy surrogate (train subset): {fe_train:.4f}")
    print(f"free-energy surrogate (test subset):  {fe_test:.4f}")

    if args.use_label_spins:
        eval_pixels_train = jnp.array(x_train_pixels[: args.eval_samples])
        eval_pixels_test = jnp.array(x_test_pixels[: args.eval_samples])
        train_preds = predict_labels(params, model.beta, eval_pixels_train, n_pixels, args.n_hidden, args.n_classes)
        test_preds = predict_labels(params, model.beta, eval_pixels_test, n_pixels, args.n_hidden, args.n_classes)
        y_train_eval = jnp.array(y_train[: args.eval_samples])
        y_test_eval = jnp.array(y_test[: args.eval_samples])
        train_acc = float(jnp.mean(train_preds == y_train_eval))
        test_acc = float(jnp.mean(test_preds == y_test_eval))
        print(f"classification accuracy (train subset, {args.eval_samples} samples): {train_acc * 100:.2f}%")
        print(f"classification accuracy (test subset, {args.eval_samples} samples):  {test_acc * 100:.2f}%")

    if args.save_params:
        save_params_npz(Path(args.save_params), params, model.beta)
        print(f"Saved parameters to {args.save_params}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="MNIST Ising EBM demo with THRML (clamped training)")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--n-hidden", type=int, default=256)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--beta", type=float, default=1.0)
    parser.add_argument("--n-warmup", type=int, default=50)
    parser.add_argument("--n-samples", type=int, default=1)
    parser.add_argument("--steps-per-sample", type=int, default=2)
    parser.add_argument("--n-pos-chains", type=int, default=1)
    parser.add_argument("--n-neg-chains", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--binarize-thresh", type=float, default=0.5)
    parser.add_argument("--data-dir", type=str, default=None)
    parser.add_argument("--eval-samples", type=int, default=1000)
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-test-samples", type=int, default=None)
    parser.add_argument("--n-classes", type=int, default=10)
    parser.add_argument(
        "--use-label-spins",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Augment visible spins with one-hot label spins and use conditional negative phase (pixels clamped).",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default=None,
        help="Directory to save per-epoch checkpoints as params_epochXXXX.npz",
    )
    parser.add_argument(
        "--load-params",
        type=str,
        default=None,
        help="Optional path to .npz checkpoint to initialize weights/biases/beta before training (or for eval-only runs).",
    )
    parser.add_argument(
        "--save-params",
        type=str,
        default=None,
        help="Optional path to .npz file where final weights/biases/beta will be saved.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    start = time.time()
    run_training(args)
    elapsed = time.time() - start
    print(f"Runtime: {elapsed/60:.2f} minutes")


if __name__ == "__main__":
    main()
