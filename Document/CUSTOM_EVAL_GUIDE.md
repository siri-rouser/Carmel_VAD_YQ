# Guide: Evaluating Fine-tuned Models on Custom Datasets with lmms-eval

This guide will help you modify the lmms-eval example to evaluate your own fine-tuned model on your custom dataset.

## Overview

The lmms-eval framework has two main components:
1. **Models** - Located in `lmms-eval/lmms_eval/models/` - Handles model inference
2. **Tasks** - Located in `lmms-eval/lmms_eval/tasks/` - Handles dataset loading and evaluation

To evaluate your custom model on your custom dataset, you need to:
1. Create/modify a model wrapper (or use an existing one like `vllm`)
2. Create a custom task YAML configuration and utilities
3. Run the evaluation with appropriate parameters

---

## Step 1: Prepare Your Model

### Option A: Using an existing model wrapper (Recommended for Quick Start)

Your current script uses the `vllm` backend with Qwen3-VL. You can adapt this for your fine-tuned model:

```bash
# The model path should point to your fine-tuned model
MODEL="/output/sft_qwen3_4b_carmel_vad/pytorch_model.bin"

# Or if your model is saved as HuggingFace format:
MODEL="/path/to/your/fine-tuned/model"
```

### Option B: Creating a custom model wrapper

If your model has special inference requirements, create a new model file:

**File: `lmms-eval/lmms_eval/models/chat/your_custom_model.py`**

```python
from lmms_eval.api.registry import register_model
from lmms_eval.models.chat.vllm import VLLM

@register_model("your_custom_model")
class YourCustomModel(VLLM):
    """Custom wrapper for your fine-tuned model"""
    
    def __init__(self, model, **kwargs):
        # Add custom initialization logic here
        super().__init__(model=model, **kwargs)
```

Then use it with: `--model your_custom_model`

---

## Step 2: Create a Custom Task

### Directory Structure

Create a new task directory for your custom dataset:

```
lmms-eval/lmms_eval/tasks/carmel_vad_custom/
├── carmel_vad_custom.yaml
└── utils.py
```

### Step 2a: Create `utils.py`

**File: `lmms-eval/lmms_eval/tasks/carmel_vad_custom/utils.py`**

```python
import json
import os
from pathlib import Path
from loguru import logger as eval_logger

# Example for image-based VAD task
def carmel_vad_doc_to_visual(doc):
    """Convert document to visual input for the model"""
    # If your dataset uses image paths
    if "image_path" in doc:
        from PIL import Image
        return [Image.open(doc["image_path"]).convert("RGB")]
    # If your dataset uses video paths
    elif "video_path" in doc:
        return [doc["video_path"]]  # vLLM handles video loading
    # If images are already loaded
    elif "image" in doc:
        return [doc["image"]]
    return []

def carmel_vad_doc_to_text(doc, lmms_eval_specific_kwargs=None):
    """Convert document to text prompt for the model"""
    pre_prompt = lmms_eval_specific_kwargs.get("pre_prompt", "") if lmms_eval_specific_kwargs else ""
    post_prompt = lmms_eval_specific_kwargs.get("post_prompt", "") if lmms_eval_specific_kwargs else ""
    
    # Adjust based on your task (e.g., anomaly detection, classification)
    question = doc.get("question", "Describe what you see in this image/video.")
    
    return f"{pre_prompt}{question}{post_prompt}"

def carmel_vad_process_results(doc, results):
    """Process model output and ground truth for metric calculation"""
    pred = results[0] if results else ""
    ground_truth = doc.get("answer", doc.get("label", ""))
    
    return {
        "carmel_vad_score": {
            "pred": pred,
            "answer": ground_truth,
            "id": doc.get("id", ""),
        }
    }

def carmel_vad_aggregate_results(results):
    """Aggregate individual results into final metrics"""
    correct = 0
    total = len(results)
    
    for result in results:
        # Simple string matching - adjust based on your metric needs
        if result["pred"].strip().lower() == result["answer"].strip().lower():
            correct += 1
    
    return {
        "accuracy": correct / total if total > 0 else 0.0,
        "correct": correct,
        "total": total,
    }
```

### Step 2b: Create `carmel_vad_custom.yaml`

**File: `lmms-eval/lmms_eval/tasks/carmel_vad_custom/carmel_vad_custom.yaml`**

#### Option 1: Using a local JSON dataset

```yaml
# Dataset Configuration
dataset_path: json  # Use json loader
dataset_kwargs:
  data_files: /path/to/your/dataset.json  # Path to your local JSON file
  split: "train"  # or "test", "val"

task: "carmel_vad_custom"
test_split: test

# Model Output Type
output_type: generate_until

# Document Processing Functions
doc_to_visual: !function utils.carmel_vad_doc_to_visual
doc_to_text: !function utils.carmel_vad_doc_to_text
doc_to_target: "answer"  # Ground truth field name

# Generation Settings
generation_kwargs:
  max_new_tokens: 256
  temperature: 0.7
  top_p: 0.9
  num_beams: 1
  do_sample: true

# Results Processing
process_results: !function utils.carmel_vad_process_results
metric_list:
  - metric: carmel_vad_score
    aggregation: !function utils.carmel_vad_aggregate_results
    higher_is_better: true

# Model-specific Prompts
lmms_eval_specific_kwargs:
  default:
    pre_prompt: ""
    post_prompt: "\nPlease provide a detailed answer."

metadata:
  - version: 0.0
```

#### Option 2: Using HuggingFace Dataset Hub

```yaml
dataset_path: your_username/your_dataset_name  # From HuggingFace Hub
dataset_kwargs:
  token: true
  cache_dir: ~/.cache/huggingface

# ... rest same as above
```

#### Option 3: Using Local Dataset with Custom Script

```yaml
# If you need custom data loading logic
dataset_path: lmms-eval/lmms_eval/tasks/carmel_vad_custom/carmel_vad_dataset.py
dataset_kwargs:
  data_dir: /path/to/your/data

# ... rest same as above
```

### Example Dataset Format (JSON)

Your JSON file should have this structure:

```json
{
  "data": [
    {
      "id": "sample_1",
      "image_path": "/path/to/image1.jpg",
      "question": "Is there an anomaly in this image?",
      "answer": "Yes, there is an anomaly in the top-left corner."
    },
    {
      "id": "sample_2",
      "image_path": "/path/to/image2.jpg",
      "question": "Describe any anomalies you see.",
      "answer": "No anomalies detected."
    }
  ]
}
```

---

## Step 3: Prepare Your Dataset

### Format Your Data

Convert your dataset to one of these formats:

1. **JSON Format** (Simplest)
```json
{
  "data": [
    {
      "id": "1",
      "image_path": "/full/path/to/image.jpg",
      "question": "Your question here",
      "answer": "Expected answer"
    }
  ]
}
```

2. **CSV Format**
Use the csv dataset loader from HuggingFace datasets

3. **HuggingFace Dataset Hub**
Push your dataset to HuggingFace Hub for easy sharing

### Update Dataset Paths

In your YAML file, ensure paths are absolute or relative to where you run the evaluation:

```yaml
dataset_kwargs:
  data_files: /home/yuqiang/yl4300/project/Carmel_VAD_YQ/your_data.json
```

---

## Step 4: Create Custom Evaluation Script

**File: `carmel_vad_eval.sh`**

```bash
#!/bin/bash

# Model Configuration
MODEL_PATH="/output/sft_qwen3_4b_carmel_vad/pytorch_model.bin"
# Or for HuggingFace model:
# MODEL_PATH="your_username/your_fine_tuned_model"

# Parallelization Settings
TENSOR_PARALLEL_SIZE=4
DATA_PARALLEL_SIZE=1

# Memory and Performance
GPU_MEMORY_UTILIZATION=0.85
BATCH_SIZE=16

# Your Custom Task
TASK="carmel_vad_custom"

# Output Configuration
OUTPUT_PATH="./logs/carmel_vad_eval"
LOG_SAMPLES=true
LOG_SUFFIX="carmel_vad_custom"

# NCCL Configuration for multi-GPU
export NCCL_BLOCKING_WAIT=1
export NCCL_TIMEOUT=18000000

echo "=========================================="
echo "Carmel VAD Custom Evaluation"
echo "=========================================="
echo "Model: $MODEL_PATH"
echo "Task: $TASK"
echo "Output Path: $OUTPUT_PATH"
echo "=========================================="

# Build and run command
CMD="uv run python -m lmms_eval \
    --model vllm \
    --model_args model=${MODEL_PATH},tensor_parallel_size=${TENSOR_PARALLEL_SIZE},data_parallel_size=${DATA_PARALLEL_SIZE},gpu_memory_utilization=${GPU_MEMORY_UTILIZATION} \
    --tasks ${TASK} \
    --batch_size ${BATCH_SIZE} \
    --output_path ${OUTPUT_PATH}"

if [ "$LOG_SAMPLES" = true ]; then
    CMD="$CMD --log_samples --log_samples_suffix ${LOG_SUFFIX}"
fi

echo "Running command:"
echo "$CMD"
echo ""

eval $CMD

echo ""
echo "=========================================="
echo "Evaluation Complete!"
echo "Results saved to: $OUTPUT_PATH"
echo "=========================================="
```

---

## Step 5: Run Evaluation

### Make the script executable
```bash
chmod +x carmel_vad_eval.sh
```

### Run the evaluation
```bash
./carmel_vad_eval.sh
```

### Or run directly
```bash
cd lmms-eval
python -m lmms_eval \
    --model vllm \
    --model_args model=/output/sft_qwen3_4b_carmel_vad/pytorch_model.bin \
    --tasks carmel_vad_custom \
    --batch_size 16 \
    --output_path ./logs/carmel_vad_eval \
    --log_samples
```

---

## Step 6: Check Results

Results will be saved in the output directory:

```
./logs/carmel_vad_eval/
├── results_<timestamp>.json       # Main results
├── samples_<timestamp>.jsonl      # Individual samples if --log_samples used
└── metrics_<timestamp>.json       # Aggregated metrics
```

---

## Common Customizations

### For Video Input
```python
def carmel_vad_doc_to_visual(doc):
    """For video files"""
    return [doc["video_path"]]  # vLLM will handle decoding
```

### For Multi-image Input
```python
def carmel_vad_doc_to_visual(doc):
    """For multiple images"""
    images = []
    for i in range(1, 5):  # 4 images per sample
        images.append(Image.open(doc[f"image_{i}_path"]))
    return images
```

### For Classification Tasks
```python
def carmel_vad_aggregate_results(results):
    """For classification with multiple classes"""
    from sklearn.metrics import accuracy_score, precision_recall_fscore_support
    
    preds = [r["pred"] for r in results]
    gts = [r["answer"] for r in results]
    
    accuracy = accuracy_score(gts, preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        gts, preds, average="weighted"
    )
    
    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }
```

### For Detection/Anomaly Tasks
```python
def carmel_vad_aggregate_results(results):
    """For detection tasks"""
    from sklearn.metrics import confusion_matrix, roc_auc_score
    
    preds = [1 if "yes" in r["pred"].lower() else 0 for r in results]
    gts = [1 if "yes" in r["answer"].lower() else 0 for r in results]
    
    tn, fp, fn, tp = confusion_matrix(gts, preds).ravel()
    
    return {
        "accuracy": (tp + tn) / (tp + tn + fp + fn),
        "precision": tp / (tp + fp) if (tp + fp) > 0 else 0,
        "recall": tp / (tp + fn) if (tp + fn) > 0 else 0,
        "f1": 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0,
    }
```

---

## Troubleshooting

### Error: "Task 'carmel_vad_custom' not found"
- Ensure the YAML file is in `lmms-eval/lmms_eval/tasks/carmel_vad_custom/carmel_vad_custom.yaml`
- Check the task name matches the YAML filename (without .yaml)

### Error: "Module not found"
- Make sure you're running from the `lmms-eval` directory
- Install lmms-eval in development mode: `pip install -e .`

### CUDA Out of Memory
- Reduce `BATCH_SIZE` in the script
- Reduce `GPU_MEMORY_UTILIZATION` (e.g., to 0.7)
- Reduce `max_new_tokens` in YAML

### Slow Evaluation
- Increase `BATCH_SIZE` if GPU memory allows
- Reduce `max_new_tokens` if not critical
- Use fewer GPUs if data transfer is bottleneck

---

## Next Steps

1. **Prepare your dataset** in JSON format
2. **Create the task directory** and files
3. **Create your evaluation script**
4. **Run a test** with `--limit 10` to check everything works
5. **Run full evaluation** on your dataset

For questions, refer to the lmms-eval documentation or check existing task implementations in `lmms_eval/tasks/`.
