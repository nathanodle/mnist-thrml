# MNIST THRML Demo

This repo packages a minimal MNIST energy-based model built with THRML and managed entirely with `uv`.  
Commands below assume you are inside `mnist-thrml/`.

## Environment via uv
1. **Create / activate venv (one time)**
   ```bash
   uv venv -p 3.11
   source .venv/bin/activate
   ```
2. **Install project in editable mode**
   ```bash
   UV_CACHE_DIR="$PWD/.uv-cache" uv pip install -e .
   ```
3. **Lock (optional but recommended for repro)**
   ```bash
   UV_CACHE_DIR="$PWD/.uv-cache" uv lock
   ```
There is no `requirements.txt`; always use `uv run …` so the locked environment is picked up automatically (activation is optional once the venv exists).

## Training
Run the main script directly with `uv` so dependencies resolve through the lock:
```bash
UV_CACHE_DIR="$PWD/.uv-cache" \
uv run python train_mnist_thrml.py \
  --checkpoint-dir checkpoints/full_run \
  --save-params checkpoints/final.npz
```
Key flags:
- `--checkpoint-dir DIR` — saves `params_epochXXXX.npz` after every epoch.
- `--save-params FILE` — dumps the final weights/biases/β (handy for later inspection).
- `--max-train-samples/--max-test-samples` — limit data for smoke tests.
- `--n-warmup`, `--steps-per-sample`, `--n-pos-chains`, `--n-neg-chains` — control Gibbs sampling effort.

## Previewing individual digits
Use the helper script to visualize a binarized sample and optionally report its free-energy surrogate:
```bash
uv run python show_sample.py \
  --split test \
  --index 42 \
  --checkpoint checkpoints/full_run/params_epoch0005.npz \
  --save-image sample42.png
```
If `--checkpoint` is omitted the script still shows the image + label; with a checkpoint it prints and overlays `free-energy surrogate = …`.

## Minimal evaluation sweep
`verbose_eval.py` (prototype CLI) estimates free energies for random samples:
```bash
uv run python verbose_eval.py \
  --checkpoint checkpoints/full_run/params_epoch0005.npz \
  --split test \
  --sample-count 20 \
  --random \
  --seed 42
```

## Tips
- Set `JAX_PLATFORMS=cuda,cpu` (defaulted in the training scripts) so JAX prefers GPU when available but still falls back to CPU.
- If CUDA wheels (e.g., `nvidia-cusparse-cu12`) are installed via pip, the training scripts auto-append their `lib/` directories to `LD_LIBRARY_PATH` so you can run without exporting it manually.
- TFDS will download MNIST to `~/tensorflow_datasets/` on first run; reuse with `--data-dir` if needed.
- When running on production-sized MNIST, expect ~40 minutes per epoch on a CPU-only Mac. Reduce `--n_hidden`, use `--max-train-samples`, or lower sampler steps for quick iteration.

## Example training run output

```
$ uv run python train_mnist_thrml.py        --checkpoint-dir checkpoints/full_run        --save-params checkpoints/final.npz
Training with 468 batches/epoch, batch size 128, n_hidden 256
epoch 1: mean proxy loss 0.257767
Saved checkpoint to checkpoints/full_run/params_epoch0001.npz
epoch 2: mean proxy loss 0.107676
Saved checkpoint to checkpoints/full_run/params_epoch0002.npz
epoch 3: mean proxy loss 0.102537
Saved checkpoint to checkpoints/full_run/params_epoch0003.npz
epoch 4: mean proxy loss 0.098284
Saved checkpoint to checkpoints/full_run/params_epoch0004.npz
epoch 5: mean proxy loss 0.098853
Saved checkpoint to checkpoints/full_run/params_epoch0005.npz
Training complete
free-energy surrogate (train subset): -1852.1560
free-energy surrogate (test subset):  -1851.2820
classification accuracy (train subset, 1000 samples): 43.10%
classification accuracy (test subset, 1000 samples):  43.90%
Saved parameters to checkpoints/final.npz
Runtime: 267.84 minutes
```
