
# Evaluating Probes: Training Reliable Activation Probes with Few Positive Examples

This codebase accompanies the paper [Training Reliable Activation Probes With a Handful of Positive Examples](https://openreview.net/forum?id=punokGGd4V) (NeurIPS 2025 MechInterp Workshop).

## TL;DR

Monitoring advanced AI for rare misalignments faces a data challenge: abundant aligned examples but only a handful of misaligned ones. We test activation probes in this "few vs. thousands" regime on spam and honesty detection tasks.

Key findings:
- Training with many negative examples is more positive-sample-efficient than balanced training for small numbers (1-10) of positive samples
- LLM upsampling can provide a performance boost equivalent to roughly doubling the number of real positive samples
- Larger models are more positive-sample-efficient to probe

## Installation

```bash
git clone <repo-url>
cd evaluating-probes
pip install -r requirements.txt
```

Create a `.env` file in the repo root with your configuration:

```bash
OPENAI_API_KEY=your_key_here  # Required for LLM upsampling experiments
EP_CPU_NJOBS=15               # Number of parallel CPU jobs
EP_GPU_NJOBS=1                # Number of parallel GPU jobs
```

## Configuration

Experiments are defined in YAML config files in `configs/`. Each config specifies:

```yaml
run_name: "spam_gemma_9b"
model_name: "google/gemma-2-9b"
device: "cuda:0"
cache_activations: True
log_file: "logs/spam_exp_gpu.log"
seeds: [42, 43, 44, 45, 46]  # Multiple seeds for variance estimation
layers: [20]                  # Which layers to extract activations from
components:
  - "resid_post"              # Residual stream after layer
```

To specify activations extraction methodology:

```yaml
activation_extraction:
  format_type: "qr"           # Options: "qr" (on-policy), "r" (off-policy instruct), "r-no-it" (off-policy non-instruct)
```

Each config contains experiments, which specify training data, evaluation data, and optional class imbalance configurations:

```yaml
experiments:
  - name: 1-spam-pred-auc
    class_names: {0: "Ham", 1: "Spam"}
    train_on: 94_better_spam
    evaluate_on:
      - 94_better_spam
      - 87_is_spam
```
The `rebuild_config` section defines how to construct imbalanced training sets:

```yaml
rebuild_config:
  increasing_spam_fixed_total:
    - {class_counts: {0: 1750, 1: 1}}   # 1750 negatives, 1 positive
    - {class_counts: {0: 1750, 1: 5}}   # 1750 negatives, 5 positives
    - {class_counts: {0: 1750, 1: 10}}  # 1750 negatives, 10 positives
```

You can also specify LLM upsampling settings:

```yaml
rebuild_config:
  llm_upsampling_experiments:
    - {llm_upsampling: True, n_real_neg: 1750, n_real_pos: 10, upsampling_factor: 5}
```

This creates 50 synthetic positives (10 real × 5 upsampling factor) in addition to the 10 real positives.

## Probe Architectures

Next, you must specify probe architectures in the main config. This project supports six probe architectures. For each probe, all hyperparameters are defined in `configs/probes.py`. Each probe (except the attention probe) supports four aggregation methods: `mean`, `max`, `last`, `softmax`

### Linear Probe (sklearn)
Standard logistic regression on aggregated activations; very fast to train.

```yaml
architectures:
  - name: "sklearn_linear_mean"
    config_name: "sklearn_linear_mean"
```

### Attention Probe (PyTorch)
Learned attention over sequence positions, then linear classification. Trains on full sequences without pre-aggregation.

```yaml
architectures:
  - name: "attention"
    config_name: "attention"
```

### SAE Probe
Uses pre-trained Sparse Autoencoders (from SAE lens or Hugging Face) to encode activations, then trains a linear classifier on selected SAE features.

```yaml
architectures:
  - name: "sae_16k_l0_408_mean"
    config_name: "sae_16k_l0_408_mean"
```

### Mass Mean Probe
Non-trainable probe that computes the direction between class means (positive minus negative).

```yaml
architectures:
  - name: "mass_mean"
    config_name: "mass_mean"
```

### Activation Similarity Probe
Non-trainable probe that classifies based on cosine similarity to class centroids.

```yaml
architectures:
  - name: "act_sim_mean"
    config_name: "act_sim_mean"
```

## Running Experiments

After you've created your config, running experiments is simple!

### Basic Usage

```bash
python -m src.main -c gemma_spam_gpu
```

This loads `configs/gemma_spam_gpu_config.yaml` and runs all specified experiments. Some key flags here are:
- `-c CONFIG`: Config file name (without `_config.yaml` suffix)
- `--rerun`: Rerun all probes from scratch, bypassing cached results

### What Happens

1. **Model check** (off-policy only): Verify the model can solve the task before probing
2. **LLM upsampling** (if configured): Generate synthetic examples
3. **Activation extraction**: Cache activations for all datasets/layers
4. **Probe training**: Train all probe architectures across seeds and class imbalance configs
5. **Evaluation**: Evaluate on validation, test, and generalization sets

Results are saved to `results/{run_name}/`.

### Example Configs

For spam detection with Gemma-2-9b:
```bash
python -m src.main -c gemma_spam_gpu
```

For honesty detection with Llama-3.3-70B:
```bash
python -m src.main -c llama_mask_gpu
```

For Qwen-3 scaling experiments:
```bash
python -m src.main -c qwen_0.6b_gpu
python -m src.main -c qwen_1.7b_gpu
python -m src.main -c qwen_4b_gpu
python -m src.main -c qwen_8b_gpu
python -m src.main -c qwen_14b_gpu
```

## Datasets

We evaluate on two binary classification tasks:

### Spam Detection (off-policy)
- `94_better_spam`: Primary spam detection dataset
- `87_is_spam`: Generalization test set

These are off-policy datasets where the model itself didn't generate the text.

### Honesty Detection (on-policy, MASK benchmark)
- `98_mask_all_honesty`: Combined honesty detection from the MASK benchmark
- `99_mask_continuations_honesty`: Continuation-based honesty
- `100_mask_disinformation_honesty`: Disinformation detection
- `101_mask_doubling_down_honesty`: Doubling-down behavior
- `102_mask_known_facts_honesty`: Known facts honesty
- `103_mask_provided_facts_honesty`: Provided facts honesty
- `104_mask_statistics_honesty`: Statistics honesty

These are on-policy datasets where probes are trained on the model's own activations when generating text.

### Adding New Datasets

Datasets are indexed in `datasets/main.csv`. To add a new dataset:
1. Add your CSV file to `datasets/original/`
2. Add a row to `datasets/main.csv` with the source filename, probe columns, and extraction method
3. Run the cleaning script to generate cleaned output
## Results Structure

```
results/{run_name}/
├── seed_42/
│   ├── 1-spam-pred-auc/
│   │   ├── trained/           # Saved probe states
│   │   ├── val_eval/          # Validation set results
│   │   ├── test_eval/         # Test set results (same dataset)
│   │   └── gen_eval/          # Generalization results (different datasets)
│   └── 2-spam-pred-auc-increasing-spam/
│       └── ...
├── seed_43/
│   └── ...
└── output.log
```

Each evaluation directory contains JSON files with metrics (AUC, accuracy, precision, recall, FPR) for each probe configuration.

## Citation

```bibtex
@inproceedings{tyagi2025probes,
  title={Training Reliable Activation Probes With a Handful of Positive Examples},
  author={Tyagi, Riya and Heimersheim, Stefan},
  booktitle={NeurIPS 2025 Workshop on Mechanistic Interpretability},
  year={2025}
}
```
