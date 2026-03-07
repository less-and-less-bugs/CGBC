This repository contains the implementation of our paper *Beyond Heuristic Prompting: A Concept-Guided Bayesian Framework for Zero-Shot Image Recognition*.

The framework consists of three main steps: **Step 1: Environment & Data Setup**, **Step 2: Concept Generation**, and **Step 3: Concept-Guided Zero-Shot Inference**.

---

## Step 1: Environment & Data Setup

Our method is built upon [Test-Time Prompt Tuning (TPT)](https://github.com/azshue/TPT) (NeurIPS 2022). Please refer to that repository for the codebase structure and data preparation.

### Environment

Create a conda environment from `requirements.txt`:

```bash
conda create --name concept --file requirements.txt
conda activate concept
```

Or install dependencies manually according to the packages listed in `requirements.txt`.

### Datasets

Follow the [TPT repository](https://github.com/azshue/TPT) for dataset download and directory structure:

1. Download all datasets to a root directory (e.g., `data/`).
2. Rename dataset directories as suggested in `${ID_to_DIRNAME}` in `./data/datautils.py`.
3. For cross-dataset evaluation, place `split_zhou_${dataset_name}.json` files under `./data/data_splits/` (see [CoOp data splits](https://github.com/KaiyangZhou/CoOp)).

**Supported datasets**: ImageNet, ImageNet-A, ImageNet-R, ImageNet-V2, ImageNet-Sketch; Flower102, DTD, OxfordPets, StanfordCars, UCF101, Caltech101, Food101, SUN397, Aircraft, EuroSAT.

---

## Step 2: Concept Generation

Concept generation produces class-specific discriminative concepts that enhance zero-shot image classification with CLIP. Instead of using a single fixed prompt (e.g., "A photo of {class}"), we enrich each class with multiple concepts in the form "A photo of {class} with {concept}" to improve distinguishability between similar classes.

### Overview

The concept generation pipeline:

1. **LLM-based generation**: A large language model (e.g., GPT-4) proposes visually discriminative concepts for each class, given the dataset context and other classes in the dataset.
2. **CLIP-based filtering**: Generated concepts are filtered by CLIP text encoder similarity to avoid concepts that are too similar to other classes or redundant with existing concepts.
3. **Batch sampling**: Concepts are generated in batches, with a *sampling window* of similar classes considered for each batch to ensure discriminative power.

### Setup

1. **API configuration**: Place your LLM API credentials in `concept_gen/api_key.txt`:
   - Line 1: your API key
   - Line 2 (optional): base URL for custom endpoints (e.g., OpenAI-compatible proxies)
   - See `concept_gen/api_key.txt.example` for the format.

2. **Dependencies**: Use the environment from **Step 1** (`requirements.txt`). Additional packages for concept generation (e.g., transformers, openai) are included.

### Usage

Run concept generation for supported datasets:

```bash
cd /path/to/project
python -m concept_gen.concept_batch_sampling
```

By default, the script processes datasets listed in `main()` and saves results to `concept_gen/batchconcepts/{dataset_name}/50_sim/results.json`.

### Key Parameters

- `target_concepts`: Number of concepts per class (default: 50)
- `sampling_window`: Number of similar classes considered when generating each batch (default: 10)
- `similarity_threshold`: CLIP similarity threshold for filtering redundant concepts (default: 0.95)
- `model_name`: LLM model (e.g., `gpt-4.1`, `gpt-4.1-2025-04-14`)

### Output Format

Results are stored as JSON:

```json
{
  "class_name_1": ["concept1", "concept2", ...],
  "class_name_2": ["concept1", "concept2", ...]
}
```

Each concept is designed to be appended to the template: `"A photo of {class} with {concept}"`.

### Supported Datasets

The concept generator supports: ImageNet, EuroSAT, Aircraft, UCF101, Cars, SUN397, Oxford Pets, DTD, Food101, Flower102, Caltech101.

---

## Step 3: Concept-Guided Zero-Shot Inference

Step 3 implements the core inference method of our paper: the concept-guided Bayesian framework for zero-shot image recognition. It uses the concepts generated in Step 2 to perform robust zero-shot classification with CLIP.

### Implementation Location

The method is implemented in:

- **`zero_shot_hc_infer_nowanbd_batch_unique_diversity.py`** — Main inference script. Key components:
  - **ConceptCLIP** (`clip/concept_clip.py`): CLIP extended with class-specific concepts
  - **concept_mad_noise**: Robust Bayesian aggregation over concepts (MAD-based noise handling)
  - **DPP sampling**: Determinantal Point Process for diverse concept subset selection
  - **Multi-prompt combining**: Combines multiple concepts per class (e.g., with `or` / `and`)

### Prerequisites

1. Complete **Step 2** to generate concepts (or use pre-generated concepts in `concept_gen/batchconcepts/{dataset}/50_sim/results.json`).
2. Prepare datasets in the expected directory structure (see `--data`).
3. **Wandb logging** (optional): Set `WANDB_API_KEY` if you use wandb for experiment tracking:
   ```bash
   export WANDB_API_KEY=your_wandb_api_key
   ```

### How to Run

Use the provided script:

```bash
cd /path/to/project
export PYTHONPATH=/path/to/project:$PYTHONPATH

# Run with default settings (concept_mad_noise + DPP sampling)
bash scripts/test_ours_concept.sh
```

Or run directly:

```bash
export PYTHONPATH=/path/to/project:$PYTHONPATH

CUDA_VISIBLE_DEVICES=0 python zero_shot_hc_infer_nowanbd_batch_unique_diversity.py \
    --test_sets SUN397/Aircraft/eurosat/Cars/Food101/Pets/Flower102/Caltech101/DTD/UCF101 \
    --sample_mode multiple \
    --combine_op or \
    --len_prompts 3 \
    --al_mode concept_mad_noise \
    --tau 1.0 \
    --lambda_threshold 2.5 \
    --sampling_times 50 \
    --num_runs 3 \
    --sampling_method dpp \
    --concept_type 50_sim \
    --max_combinations 500 \
    --resolution 224 \
    --result_path results_report/results_concept.json
```

### Key Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--al_mode` | Aggregation algorithm: `concept_mad_noise`, `concept_avg`, `concept_map`, etc. | `concept_avg` |
| `--tau` | Temperature for concept aggregation | 1.0 |
| `--lambda_threshold` | MAD threshold for `concept_mad_noise` | 2.5 |
| `--sample_mode` | `single` or `multiple` prompts per class | `single` |
| `--combine_op` | How to combine prompts: `or`, `and`, or `,` | `or` |
| `--len_prompts` | Number of concepts combined per prompt | 2 |
| `--sampling_method` | `no` (random), `dpp`, or `brute_force` | `no` |
| `--sampling_times` | Number of concept subsets to sample | 50 |
| `--concept_type` | Concept folder name (e.g., `50_sim`) | `50_sim` |
| `--test_sets` | Datasets to evaluate (slash-separated) | — |

### Output

Results are written to the path specified by `--result_path` (e.g., `results_report/results_concept.json`).
