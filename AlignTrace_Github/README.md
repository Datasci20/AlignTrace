# AlignTrace

AlignTrace reconstructs a DNA sequence from noisy reads and a known target length. This repository contains the reconstruction pipeline, pretrained generator and reranker checkpoints, an optional learned read filter, evaluation tools, and scripts for reproducing the AlignTrace benchmark results.

## Reconstruction workflow

1. **Optional count shortcut.** Return frequent observed reads of the target length when exact-read support is sufficient.
2. **Optional learned read filter.** Identify clusters with conflicting read support and retain supported reads.
3. **Generator.** Choose a read-derived anchor, align the reads into a slot lattice, and use a convolutional model and length-aware decoder to propose 20 distinct sequences.
4. **Reranker.** Score the proposals using generator scores and read support, returning the best five.

The two method switches are `--shortcut-mode` and `--filter`, both `off` by default. They can be enabled independently. `--batch-size` controls execution and defaults to 64. Target length is supplied task information.

## Installation

Use Python 3.12. Run these commands from the repository root:

```sh
python -m venv .venv
```

Activate the environment in your shell:

```powershell
# Windows PowerShell
.\.venv\Scripts\Activate.ps1
```

```sh
# Linux/macOS
. .venv/bin/activate
```

Then install the pinned dependencies:

```sh
python -m pip install -r requirements.txt
```

For the reference PyTorch CUDA 12.8 build, install this wheel before installing the requirements:

```sh
python -m pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cu128
```

CUDA execution requires a compatible NVIDIA GPU and driver. Select `--device cpu` for CPU inference. The recorded experimental environment is in `protocol/PROTOCOL.json`.

## Predict and evaluate

The bundled demo needs no external dataset:

```sh
python predict.py --input examples/demo.jsonl --output results/demo_off.jsonl --device cuda
python evaluate.py --input examples/demo.jsonl --predictions results/demo_off.jsonl --output results/demo_accuracy.json
python predict.py --input examples/demo.jsonl --output results/demo_on.jsonl --filter on --device cuda
```

The demo illustrates the interface; it is too small to measure benchmark accuracy. Choose fresh prediction and evaluation output filenames for each run. The first inference call includes Numba compilation and, on CUDA, graph capture.

For the full Grass dataset with the paper's count shortcut:

```sh
python predict.py --input "../Real Datasets/Grass.txt" --length 117 --output results/grass.jsonl --shortcut-mode on --device cuda
python evaluate.py --input "../Real Datasets/Grass.txt" --length 117 --predictions results/grass.jsonl --output results/grass_accuracy.json
```

### Method settings

| Setting | Value | Meaning |
|---|---|---|
| `--shortcut-mode` | `off` or `on` | Enable exact-read counting before neural reconstruction |
| `--filter` | `off` or `on` | Enable learned filtering for clusters reaching neural reconstruction |
| `--batch-size` | 64 by default | Clusters per inference batch |
| Candidate budget / beam width | 20 / 24 | Fixed decoder settings |
| Returned candidates | Up to 5 | Ranked, distinct sequences of the known target length |

The shortcut triggers when the most frequent target-length read occurs at least three times or accounts for at least 60% of all reads. It returns up to five observed target-length strings ranked by count; ties follow their first occurrence in the input.

The learned filter uses a 145-parameter cluster gate with threshold **0.5**, followed by 5-mer read selection at **0.8 times the median support**. These calibration constants are fixed for the released checkpoint. Clusters with fewer than four reads are retained. Selection must preserve at least two reads and at least half the cluster; otherwise the entire cluster is retained. Gate features use upper median order statistics, while read selection uses the ordinary median.

The generator has 88,006 parameters and the reranker has 1,282. Feature ordering, architecture, and calibration settings are part of the supplied checkpoints. The Python interface is `Pipeline(filter_on=False, shortcut_on=False, batch_size=64, device='cuda', generator=None, reranker=None)`, followed by `predict(read_sets, lengths)` and `close()`. Compatible checkpoint files can also be supplied through `--generator` and `--reranker`.

### Input and output format

JSONL supports unlabeled inputs and mixed target lengths, with one cluster per line:

```json
{"id":"example","target_length":12,"reads":["ACGTACGTACGT","ACGTACGTTCGT"]}
```

Add a `reference` field for evaluation. Binned TXT input contains blank-line-separated reference/asterisk/read blocks and requires `--length`. Prediction discards reference bases and donor labels. Reads must be nonempty uppercase A/C/G/T strings of at most 4096 bases; target lengths must be integers from 3 to 4096. The alignment lattice must contain enough slots for the requested output. Length-140 inputs require at least one read of at most 180 bases after filtering, matching the released anchor policy. Empty clusters return no candidates and count as failures in every evaluation denominator.

Prediction records contain IDs, target lengths, ranked `candidates`, retained-read indices, and filter/shortcut diagnostics. The first candidate is the primary prediction. A `.gz` suffix enables compressed input/output. An adjacent `.receipt.json` records checkpoint and input hashes, fixed settings, and timings. `timing.wall` includes first-use compilation; stage times overlap because batches are pipelined.

Evaluation checks dataset order, IDs, target lengths, and candidate validity, then reports exact recovery, raw unit-cost edit distance, within-one-edit recovery, and the best result among up to five returned candidates. Short shortcut lists are scored as returned.

## Reproduce the paper's AlignTrace results

Place the companion datasets alongside the repository, or point `--datasets-root` to their parent:

```text
project/
  AlignTrace_Github/
  Real Datasets/
  Synthetic Datasets/
```

`protocol/DATASETS.json` specifies the required source files, SHA256 hashes, cluster order, and settings. The dataset archives must accompany the release; a public download URL has not yet been added. Supplied data are used directly. Synthetic dataset generation is provided in the companion package's `generation/` directory.

```sh
python reproduce.py --list-datasets
python reproduce.py --cohort real --dataset grass --datasets-root .. --output ../grass_smoke --smoke --device cuda
python reproduce.py --dataset contamination_sweep__iid_low_error_p10 --datasets-root .. --output ../filter_smoke --smoke --device cuda
python reproduce.py --datasets-root .. --output ../AlignTrace_Reproduced --device cuda
```

`--prepare-only` verifies source hashes and prepares selected inputs without inference. `--smoke` evaluates the first 32 clusters of the first selected dataset with one warm repetition; use a separate output directory. Full benchmarks use batch 64 and the mean of three synchronized warm calls after an initial call for compilation/capture.

| Cohort | Dataset settings | Model and filter |
|---|---:|---|
| `real` | 6 | Synthetic-trained checkpoints; filter off |
| `real_test75` | 6 | Synthetic-trained and real-selected checkpoints on frozen held-out subsets; filter off |
| `synthetic` | 39 | Synthetic-trained checkpoints; filter off |
| `contamination_sweep` | 42 | Seven channels at 0%, 10%, 20%, 30%, 40%, and 50%; filter off/on |

The full run contains **93 dataset settings and 141 model/dataset comparisons**. It evaluates 346,080 full-real, 259,615 held-out-real, 67,320 synthetic, and 25,200 contamination cluster instances. These groups reuse targets and must not be summed as independent sequences. The shortcut is enabled only for Grass and Illumina in the real cohorts. Every supplied read is used unless the learned filter is enabled.

The frozen held-out split is grouped by canonical sequence or indexed payload. Chandak and Illumina retain synthetic weights after development selection; FirstFlowcell retains generator adaptation; Trellis, Grass, and SecondFlowcell retain joint updates. `protocol/FINETUNED_MODELS.json` maps the selected checkpoints. Training and external baseline implementations are outside this repository's reproduction scope.

Contamination retains all genuine reads and appends noisy donor reads from other references in the same channel/profile. At genuine depth `n` and requested foreign fraction `p`, the added count is `floor(n*p/(1-p)+u)` for a fixed uniform draw `u`. Levels share targets and nested donor streams. Preparation verifies the frozen content hashes; donor labels never enter inference.

Outputs include:

- Prepared `data/`, ranked `predictions/`, and per-dataset `results/` with hashes and timing receipts.
- `dataset_metrics.csv` and `COMPARISON.json`, comparing new metrics and candidate digests with the saved reference run.
- `TABLE_SUMMARIES.json`, containing cluster-weighted cohort/contamination aggregates and paired burst8/burst10 depth analysis.
- `stratified_metrics.csv`, containing accuracy by target length, exact read depth, and depth below/equal to/above 10. These supply the length and depth summaries in the supplement.
- `ENVIRONMENT.json` and `RUN_CONFIG.json`, recording the execution environment and protocol fingerprints.

Partial selections produce summaries for their listed comparisons only. Resume a run with the same output directory and unchanged code, data, and settings. Resume checks also verify saved prediction hashes. Runtime ratios to the reference GPU run are reported only for CUDA calls with verified GPU isolation; hardware and software differences still affect runtimes and numerical ties.

## Verification and attribution

The regression suite can be run in an installed environment:

```sh
python -m unittest discover -s tests -v
```

The regression suite checks input validation, separation of reference labels from prediction, checkpoint integrity, dataset selection, and benchmark aggregation. Protocol files preserve the input hashes, settings, expected results, and model mappings required to reproduce the benchmark.

AlignTrace's original code and included pretrained model weights are licensed under the [MIT License](LICENSE.md). The same file contains third-party notices; the complete retained third-party license texts are in `licenses/`. Companion datasets have separate reuse terms.
