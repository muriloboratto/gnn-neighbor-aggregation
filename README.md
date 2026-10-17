# GNN Neighbor Aggregation: CUDA C++ vs. PyTorch

A small, hands-on Google Colab experiment comparing a custom **CUDA C++ neighbor-aggregation kernel** with **PyTorch sparse CSR** and (for smaller graphs) **PyTorch dense** matrix multiplication. Designed as an introductory GPU programming and performance-analysis lab.

## Learning objectives

- Understand graph neighbor aggregation and **Compressed Sparse Row (CSR)** storage.
- Implement the forward operation using a CUDA C++ kernel.
- Call a compiled CUDA shared library from PyTorch using `ctypes` and GPU tensor pointers.
- Validate numerical correctness and benchmark GPU execution using CUDA events.
- Discuss how graph size, degree, and feature dimension affect performance.

## Notebook

**File:** `gnn_cuda_colab_english.ipynb`

Open [Google Colab](https://colab.research.google.com/drive/1jWKMcAluoJmGBcdiC7yMthSXgLc24VIT?usp=sharing), upload the notebook, then choose **Runtime → Change runtime type → GPU**. Run the cells in order.

### Requirements

- A Google Colab runtime with an NVIDIA GPU and the **CUDA Toolkit (`nvcc`)** available.
- Python 3 with **PyTorch (CUDA-enabled)**, NumPy, pandas, and Matplotlib.

The first executable cell checks GPU availability and `nvcc`, installs any missing Python packages with `pip`, and prints the environment details. **Note:** installing `torch` alone does not guarantee a CUDA-enabled build; the notebook checks `torch.cuda.is_available()` and stops if a usable GPU is missing. If `nvcc` is absent, choose a runtime with the CUDA Toolkit installed.

## Mathematical operation

For a graph with `N` nodes and `F` features per node, the aggregation computes:

$$
Y = AX, \qquad A\in\mathbb{R}^{N\times N},\quad X,Y\in\mathbb{R}^{N\times F}.
$$

In CSR form, the output for node `i` and feature `f` is:

$$
Y_{i,f}=\sum_{e=\mathrm{row\_ptr}[i]}^{\mathrm{row\_ptr}[i+1]-1}
\mathrm{values}[e]\,X_{\mathrm{col\_idx}[e],f}.
$$

CSR uses three arrays:

| Array | Meaning |
| --- | --- |
| `row_ptr` | Start/end offsets of each node's neighbor list |
| `col_idx` | Neighbor node indices |
| `values` | Edge weights |

The notebook generates synthetic graphs with a **fixed number of sampled neighbors per node**. Each edge weight is `1 / degree`; the resulting aggregation is a simple mean over the sampled entries. Duplicate neighbor indices can occur because neighbors are sampled with replacement. This is an educational aggregation benchmark, **not a complete GNN training pipeline**.

## Implementations compared

| Implementation | Description | Used for |
| --- | --- | --- |
| PyTorch sparse CSR | `torch.sparse.mm(csr, x)` | Sparse reference and baseline |
| Custom CUDA C++ | One thread computes one `(node, feature)` output, looping over that node's CSR entries | Custom-kernel baseline |
| PyTorch dense | `dense @ x` | Small-graph comparison only (`N <= 2048`) |

The notebook writes `neighbor_aggregate.cu` and compiles `libneighbor_aggregate.so` using:

```bash
nvcc -O3 -std=c++17 -shared -Xcompiler -fPIC \
  neighbor_aggregate.cu -o libneighbor_aggregate.so
```

The Python code invokes the shared library through `ctypes`, passing pointers to tensors already on the GPU and the current CUDA stream. It does not copy the tensors back to the CPU for each aggregation.

## Experiment configuration

The initial correctness and timing test uses:

| Parameter | Value |
| --- | --- |
| Nodes | 2048 |
| Neighbors per node | 16 |
| Features per node | 64 |
| Warm-up iterations | 20 |
| Timed repetitions | 100 |

The size sweep uses `N = [512, 1024, 2048, 4096, 8192]`, with 16 neighbors per node, 64 features, and 100 repetitions. Dense PyTorch is measured only up to 2048 nodes to limit quadratic memory use.

### Correctness

The custom CUDA output is compared with PyTorch CSR using:

```python
torch.testing.assert_close(custom_output, reference, rtol=1e-4, atol=1e-4)
```

The dense output is also checked where enabled.

### Timing and speedup

GPU timings use `torch.cuda.Event` after warm-up. The reported time is the average elapsed time per call over the repetitions, including the Python-side launch path and GPU work, but **excluding graph generation, compilation, and host-to-device setup**.

$$
\mathrm{Speedup}=\frac{T_{\mathrm{PyTorch\ CSR}}}{T_{\mathrm{custom\ CUDA}}}.
$$

- **Speedup > 1:** the custom CUDA path is faster in that measurement.
- **Speedup < 1:** the PyTorch CSR path is faster.

Results depend on the Colab GPU, CUDA/PyTorch versions, GPU clocks, contention, and graph shape. Do not interpret a single run as a general performance ranking.

## Outputs

- Printed environment and CUDA compilation details.
- Correctness confirmation and timings for the initial test.
- A pandas table covering all graph sizes.
- A plot comparing PyTorch CSR, custom CUDA, and PyTorch dense where applicable.
- **`gnn_cuda_results.csv`**, saved in the notebook's working directory.

In Colab, the CSV can be downloaded by uncommenting the provided `google.colab.files.download(...)` lines.

## Suggested lab questions

1. What do `row_ptr`, `col_idx`, and `values` represent in CSR?
2. Why is each CUDA thread mapped to one `(node, feature)` output element?
3. How does the feature dimension influence parallelism and memory access?
4. Why is dense adjacency expensive for large, sparse graphs?
5. Under which conditions does the custom kernel outperform PyTorch CSR?
6. How would you redesign the kernel to support irregular node degrees or warp-cooperative aggregation?

## Scope and limitations

- **Forward aggregation only:** no trainable weights, loss function, backpropagation, or end-to-end GNN training.
- **Single GPU:** no NCCL, MPI, or multi-node execution.
- **Synthetic fixed-degree graphs:** not a real-world graph dataset.
- **Baseline kernel:** no warp-level cooperative reductions, degree-aware scheduling, or kernel fusion.
- **GPU required:** the notebook is not intended to run its CUDA benchmark on a CPU-only Colab runtime.

Potential extensions include a warp-per-node CUDA kernel, variable-degree graphs, neighbor sampling, a backward pass, and multi-GPU graph partitioning.
