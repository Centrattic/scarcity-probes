# Datasets

Datasets sourced from a variety of locations (most credit to [SAE-Probes](https://github.com/JoshEngels/SAE-Probes/blob/main/data/probing_datasets_MASTER.csv)), processed through our cleaning pipeline.

## Structure

```
datasets/
├── main.csv          # Index of all datasets
├── cleaning.py       # Run this to process datasets
├── original/         # Raw source files
├── cleaned/          # Processed output files
└── handlers/         # Processing handlers
```

## Output Format

All handlers produce CSVs with three columns:
- `prompt`: The text to extract activations from
- `prompt_len`: Character length of prompt
- `target`: Label (binary 0/1 or categorical)

## Adding New Datasets

1. Place your source file in `original/`
2. Add a row to `main.csv` with: `number`, `save_name`, `source`, `Probe from`, `Probe to`, extraction expressions, and `handler`
3. Run `python cleaning.py`

## Handlers

Handlers are specified in the `handler` column of `main.csv`. Each handler has a `process(row, source_file)` function.

### simple
General-purpose handler for tabular data. Uses Python expressions from `probe from extraction` and `probe to extraction` to transform columns specified in `Probe from` and `Probe to`.

### generic_mcq_balancer
Balances multiple-choice datasets 50/50 correct/incorrect. `Probe from` specifies question column + distractor columns; `Probe to` specifies correct answer column. Outputs `Q: {question} A: {answer}` with binary correctness label.

### spam_handler
Extends `simple` with spam-specific preprocessing: robust CSV parsing for malformed emails, filters top 20% longest samples, and samples up to N per class for balance.

### mask_metrics_handler
Processes MASK benchmark honesty data. Combines multiple source CSVs, extracts model generations, and maps honesty scores (-1=lie→1, 1=truth→0).

### hf_handler
Meta-handler for Hugging Face datasets. Syntax: `hf_handler[downstream_handler]`. Downloads from HF, then passes to the specified downstream handler.

### Other Handlers
- `choices_dict_label` / `choices_list_label`: MCQ with choices as dict/list
- `arithmetic_label`: MCQ with labeled options (A, B, C, D)
- `phys_reasoning_balancer`: Physics reasoning with separate label files
- `science_commonsense_merge`: Merges train/test/validation splits
- `eng_french_balancer` / `translate_french_mcq_balancer`: Translation tasks
- `text_info_label`: Text classification with info extraction
- `llama_combined_handler`: Combines multiple Llama-specific sources
- `context_reasoning_rowwise`: Row-wise context reasoning extraction
