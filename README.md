# Anonymous supplementary code

This archive contains the GPU kernel and a memory-bounded example for fitting
an isotropic Gaussian mixture model to one billion 128-dimensional vectors.
It is provided for anonymous review and intentionally contains no author or
repository metadata.

## Contents

- `flash_gmm.py`: the source-hashed mixed-precision Triton kernel recovered
  from the one-billion-vector benchmark workspace and used in the numerical
  validation rerun.
- `sift1b_chunked_example.py`: an end-to-end streamed EM example for the
  standard SIFT1B `u8bin` representation.
- `LICENSE`: software license.

The SHA-256 digest of `flash_gmm.py` is:

```text
c9dc984117413cd968cfdfde5d7e86b3044698b172cc002c0ca50908b4a82e60
```

## Requirements

- Linux with a CUDA-capable GPU
- Python 3.10 or newer
- PyTorch with CUDA support
- Triton
- NumPy

Install the Python dependencies in an appropriate CUDA environment:

```bash
python -m pip install numpy torch triton
```

The reported large-scale configuration used an NVIDIA H100 80 GB GPU. The
kernel selects TF32 or centered BF16 according to the rows-per-component
ratio; its default launch configuration is tuned for H100-class hardware.
The retained validation environment used Python 3.13.14, PyTorch 2.9.1 with
CUDA 12.8, Triton 3.5.1, and NumPy 2.4.4.

## Input files

The example expects:

1. A `u8bin` dataset whose first eight bytes are two little-endian unsigned
   32-bit integers `(N, D)`, followed by `N * D` row-major unsigned bytes.
2. A `.npy` array of initial centers, or an `.npz` file containing the centers.
   For the default settings the center array must have shape `(10000, 128)`.

For an `.npz` file, the default key is `centroids_10`; override it with
`--init-centers-key` when necessary. To reproduce the warm-start protocol,
provide centroids produced by ten iterations of k-means on the same dataset.

Datasets and trained parameters are not included in this archive.

## One-billion-vector example

From the archive directory, run:

```bash
python sift1b_chunked_example.py \
  --data /path/to/base.1B.u8bin \
  --init-centers /path/to/kmeans_centers.npz \
  --components 10000 \
  --rows 1000000000 \
  --chunk-rows 10000000 \
  --iterations 30 \
  --output sift1b_run.json \
  --save-parameters sift1b_parameters.pt
```

The defaults for the number of components, chunk size, and EM iterations are
the retained one-billion-vector configuration. The example first performs a
streamed hard-assignment pass to initialize mixture weights and spherical
variances from the supplied centers. It then runs each EM iteration as follows:

1. Stage each input chunk through pinned host memory.
2. Evaluate the unchanged Flash-GMM E-step kernel on that chunk.
3. Accumulate counts, first moments, scatter, and log likelihood in FP64.
4. Apply one M-step only after all chunks have been processed.

This preserves the EM semantics: model parameters are held fixed throughout
all chunks belonging to one iteration. The full input and the `N x K`
responsibility matrix are never materialized on the GPU.

For a smaller smoke test, use the same file and initialization with a reduced
row count and chunk size, for example:

```bash
python sift1b_chunked_example.py \
  --data /path/to/base.1B.u8bin \
  --init-centers /path/to/kmeans_centers.npz \
  --rows 1000000 \
  --components 10000 \
  --chunk-rows 100000 \
  --iterations 1
```

## Important numerical detail

The kernel returns scatter around the means supplied to the E-step. These
statistics are additive across chunks when the same means are used for every
chunk. The supplied `flash_gmm_mstep` function applies the mean-shift
correction before computing the updated spherical variances. Do not divide the
raw scatter by `D * Nk` without this correction.

Chunking and atomic accumulation can change floating-point reduction order;
bitwise equality between chunked and unchunked execution is not expected.
