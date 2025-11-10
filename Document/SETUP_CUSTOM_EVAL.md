# Carmel VAD - Custom Model Evaluation Setup

This guide walks you through evaluating your fine-tuned Carmel VAD model on your custom dataset using lmms-eval.

## Quick Start (5 minutes)

### 1. Prepare Your Dataset

Your dataset should be in JSON format:

```json
{
  "data": [
    {
      "id": "sample_001",
      "image_path": "/full/path/to/image.jpg",
      "question": "Is there an anomaly?",
      "answer": "Yes, there is an anomaly",
      "category": "anomaly_detection"
    }
  ]
}
```

See `dataset_example.json` for more examples.

### 2. Update Task Configuration

Edit `lmms-eval/lmms_eval/tasks/carmel_vad_custom/carmel_vad_custom.yaml`:

```yaml
dataset_path: json
dataset_kwargs:
  data_files: /path/to/your/dataset.json
  split: "train"
```

Replace `/path/to/your/dataset.json` with your actual dataset path.

### 3. Run Evaluation

```bash
cd lmms-eval

# Test with 10 samples first
bash ../examples/models/carmel_vad_test.sh

# Run full evaluation
bash ../examples/models/carmel_vad_custom.sh
```

Check results in `logs/carmel_vad_eval/`

---

## Step-by-Step Guide

### Step 1: Prepare Your Dataset

#### Format Requirements

Your dataset **must** be in JSON format with the following structure:

```json
{
  "data": [
    {
      "id": "unique_sample_id",
      "image_path": "/absolute/path/to/image.jpg",
      "question": "Your question here",
      "answer": "Expected answer",
      "category": "optional_category_name"
    }
  ]
}
```

#### Supported Input Types

| Field | Type | Example | Support |
|-------|------|---------|---------|
| `image_path` | String | `/path/to/image.jpg` | ✓ Single image |
| `video_path` | String | `/path/to/video.mp4` | ✓ Video files |
| `image` | PIL Image | Image object | ✓ Pre-loaded |
| `image_paths` | List | `["/path/1.jpg", "/path/2.jpg"]` | ✓ Multiple images |

#### Example Datasets

**Simple Anomaly Detection:**
```json
{
  "data": [
    {
      "id": "1",
      "image_path": "/data/images/sample1.jpg",
      "question": "Is there an anomaly?",
      "answer": "Yes"
    },
    {
      "id": "2",
      "image_path": "/data/images/sample2.jpg",
      "question": "Is there an anomaly?",
      "answer": "No"
    }
  ]
}
```

**Video Understanding:**
```json
{
  "data": [
    {
      "id": "video1",
      "video_path": "/data/videos/sample1.mp4",
      "question": "Describe what happens in this video",
      "answer": "A person walks across the room"
    }
  ]
}
```

**Multi-image Input:**
```json
{
  "data": [
    {
      "id": "multi1",
      "image_paths": ["/data/frame1.jpg", "/data/frame2.jpg", "/data/frame3.jpg"],
      "question": "Describe the sequence of events",
      "answer": "The person enters from the left and exits to the right"
    }
  ]
}
```

### Step 2: Convert Your Data

If your data is currently in another format:

**From CSV to JSON:**
```python
import json
import pandas as pd

df = pd.read_csv('your_data.csv')
data = []
for _, row in df.iterrows():
    data.append({
        "id": row['id'],
        "image_path": row['image_path'],
        "question": row['question'],
        "answer": row['answer']
    })

with open('dataset.json', 'w') as f:
    json.dump({"data": data}, f, indent=2)
```

**From Directory Structure:**
```python
import json
import os
from pathlib import Path

dataset = []
data_dir = "/path/to/your/data"

for i, image_name in enumerate(sorted(os.listdir(data_dir))):
    if image_name.endswith('.jpg'):
        dataset.append({
            "id": f"sample_{i:05d}",
            "image_path": os.path.join(data_dir, image_name),
            "question": "Describe any anomalies in this image",
            "answer": "Your ground truth here"
        })

with open('dataset.json', 'w') as f:
    json.dump({"data": dataset}, f, indent=2)
```

### Step 3: Update Task Configuration

Edit `lmms-eval/lmms_eval/tasks/carmel_vad_custom/carmel_vad_custom.yaml`:

1. Update the dataset path:
```yaml
dataset_path: json
dataset_kwargs:
  data_files: /home/yuqiang/yl4300/project/Carmel_VAD_YQ/your_dataset.json
  split: "train"
```

2. (Optional) Customize generation parameters:
```yaml
generation_kwargs:
  max_new_tokens: 256      # Increase for longer answers
  temperature: 0.7          # Adjust randomness (0=deterministic, 1=random)
  do_sample: true           # Set to false for deterministic results
```

3. (Optional) Choose the right metric for your task:

**For Binary Classification (Yes/No):**
```yaml
metric_list:
  - metric: carmel_vad_score
    aggregation: !function utils.carmel_vad_aggregate_results_anomaly
    higher_is_better: true
```

**For Multi-class Classification:**
```yaml
metric_list:
  - metric: carmel_vad_score
    aggregation: !function utils.carmel_vad_aggregate_results_classification
    higher_is_better: true
```

### Step 4: Update Evaluation Script

Edit `lmms-eval/examples/models/carmel_vad_custom.sh`:

```bash
# Update these variables
MODEL="/output/sft_qwen3_4b_carmel_vad/pytorch_model.bin"
TENSOR_PARALLEL_SIZE=4        # Your number of GPUs
BATCH_SIZE=16                  # Adjust based on GPU memory
TASK="carmel_vad_custom"       # Keep as is
OUTPUT_PATH="./logs/carmel_vad_eval"
```

### Step 5: Test Your Setup

Before running full evaluation, test with 10 samples:

```bash
cd lmms-eval
bash ../examples/models/carmel_vad_test.sh
```

This will:
- Load your model
- Process 10 samples
- Generate predictions
- Save results to `logs/carmel_vad_test/`

Check the output file:
```bash
cat logs/carmel_vad_test/results_*.json
```

### Step 6: Run Full Evaluation

Once testing succeeds:

```bash
bash ../examples/models/carmel_vad_custom.sh
```

Results will be saved to `logs/carmel_vad_eval/`

---

## Understanding Results

After evaluation, check the output directory:

```
logs/carmel_vad_eval/
├── results_20250105_120000.json        # Main results with all metrics
├── samples_20250105_120000.jsonl       # Individual predictions (if --log_samples used)
└── metrics_summary.txt                 # Metrics summary
```

### Results File Format

```json
{
  "results": {
    "carmel_vad_custom": {
      "accuracy": 0.85,
      "correct": 425,
      "total": 500
    }
  },
  "configs": {
    "model": "/output/sft_qwen3_4b_carmel_vad/pytorch_model.bin",
    "task": "carmel_vad_custom",
    "batch_size": 16
  },
  "timestamp": "2025-01-05T12:00:00"
}
```

### Samples File Format

```jsonl
{"id": "sample_001", "pred": "Yes, there is an anomaly", "answer": "Yes, there is an anomaly", "category": "anomaly_detection", "score": 1.0}
{"id": "sample_002", "pred": "No anomalies", "answer": "No anomalies detected", "category": "anomaly_detection", "score": 0.5}
```

---

## Customization Guide

### Custom Metrics

Edit `lmms-eval/lmms_eval/tasks/carmel_vad_custom/utils.py`:

```python
def carmel_vad_aggregate_results(results):
    """Your custom metric logic"""
    # Process results
    # Return dictionary with your metrics
    return {
        "your_metric": value,
        "another_metric": value,
    }
```

Then add to YAML:
```yaml
metric_list:
  - metric: your_metric
    aggregation: !function utils.carmel_vad_aggregate_results
    higher_is_better: true
```

### Custom Input Processing

If your data has special preprocessing needs, edit `doc_to_visual`:

```python
def carmel_vad_doc_to_visual(doc):
    """Custom video preprocessing"""
    from pathlib import Path
    
    if "video_path" in doc:
        video_path = doc["video_path"]
        # Custom preprocessing here
        return [video_path]  # Return for vLLM processing
```

### Custom Prompts

Update prompts in the YAML file:

```yaml
lmms_eval_specific_kwargs:
  default:
    pre_prompt: "You are an anomaly detection expert. "
    post_prompt: "\nProvide a brief but informative response."
```

Or per-model:
```yaml
  qwen_vl:
    pre_prompt: "Analyze this image: "
    post_prompt: "\nAnswer concisely."
```

---

## Troubleshooting

### Error: "Task 'carmel_vad_custom' not found"

**Solution:**
- Verify YAML file exists: `lmms-eval/lmms_eval/tasks/carmel_vad_custom/carmel_vad_custom.yaml`
- Check file naming matches task name
- Run from `lmms-eval` directory

### Error: "File not found" for dataset

**Solution:**
- Use **absolute paths** in YAML:
  ```yaml
  data_files: /home/yuqiang/yl4300/project/Carmel_VAD_YQ/dataset.json
  ```
- Check path exists: `ls -la /path/to/dataset.json`

### Error: "CUDA out of memory"

**Solution in script:**
```bash
BATCH_SIZE=8              # Reduce from 16
GPU_MEMORY_UTILIZATION=0.7 # Reduce from 0.85
TENSOR_PARALLEL_SIZE=2    # Use fewer GPUs
```

Or in YAML:
```yaml
generation_kwargs:
  max_new_tokens: 128   # Reduce from 256
```

### Error: "Model loading failed"

**Solution:**
- Check model path is correct
- Verify model format (bin, safetensors, or HuggingFace repo)
- Ensure vLLM supports your model: `pip install vllm>=0.11.0`

### Slow Evaluation

**Solutions:**
- Increase batch size if GPU memory allows
- Reduce `max_new_tokens`
- Use more GPUs for tensor parallelism
- Profile with smaller `--limit 100`

### Evaluation Stuck or Hanging

**Solution:**
```bash
export NCCL_BLOCKING_WAIT=1
export NCCL_TIMEOUT=36000000
bash examples/models/carmel_vad_custom.sh
```

---

## Advanced Usage

### Evaluating Multiple Splits

Create separate YAML files for each split:
- `carmel_vad_custom_train.yaml` (dataset_kwargs split: train)
- `carmel_vad_custom_val.yaml` (dataset_kwargs split: val)
- `carmel_vad_custom_test.yaml` (dataset_kwargs split: test)

Run evaluation for each:
```bash
python -m lmms_eval --model vllm --model_args ... --tasks carmel_vad_custom_train carmel_vad_custom_val carmel_vad_custom_test
```

### Batch Multiple Models

```bash
for model in model1 model2 model3; do
    sed "s|^MODEL=.*|MODEL=$model|" examples/models/carmel_vad_custom.sh | bash
done
```

### Using Different Model Backends

**Using SGLang (faster for Qwen):**
Create `lmms_eval_sglang.sh`:
```bash
python -m lmms_eval \
    --model sglang \
    --model_args model=/path/to/model \
    --tasks carmel_vad_custom \
    --output_path ./logs/carmel_vad_sglang
```

**Using OpenAI-compatible API:**
```bash
python -m lmms_eval \
    --model openai_compatible \
    --model_args model_id=your-model \
    --tasks carmel_vad_custom
```

---

## File Structure

After setup, your project should look like:

```
Carmel_VAD_YQ/
├── lmms-eval/
│   ├── lmms_eval/
│   │   └── tasks/
│   │       └── carmel_vad_custom/
│   │           ├── utils.py              # Processing functions
│   │           └── carmel_vad_custom.yaml # Configuration
│   └── examples/
│       └── models/
│           ├── carmel_vad_custom.sh      # Main evaluation script
│           └── carmel_vad_test.sh        # Test script
├── dataset_example.json                  # Example dataset format
├── your_dataset.json                     # YOUR DATASET
├── CUSTOM_EVAL_GUIDE.md                  # Detailed guide
└── logs/
    └── carmel_vad_eval/
        └── results_*.json                # Results
```

---

## Next Steps

1. ✅ Prepare your dataset in JSON format
2. ✅ Update `carmel_vad_custom.yaml` with dataset path
3. ✅ Run test: `bash examples/models/carmel_vad_test.sh`
4. ✅ Run full evaluation: `bash examples/models/carmel_vad_custom.sh`
5. ✅ Analyze results in `logs/carmel_vad_eval/`
6. ✅ Customize metrics/prompts as needed

For more details, see `CUSTOM_EVAL_GUIDE.md`.
