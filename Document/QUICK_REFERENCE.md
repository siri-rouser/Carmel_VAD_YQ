# Carmel VAD Custom Model Evaluation - Quick Reference

## What's Been Created

I've set up a complete custom evaluation framework for you. Here's what was created:

### 📂 New Files Created

1. **`lmms-eval/lmms_eval/tasks/carmel_vad_custom/`** - Your custom task
   - `utils.py` - Processing functions for data and results
   - `carmel_vad_custom.yaml` - Task configuration

2. **`lmms-eval/examples/models/`** - Evaluation scripts
   - `carmel_vad_custom.sh` - Full evaluation script
   - `carmel_vad_test.sh` - Quick test script (10 samples)

3. **Root directory documentation**
   - `SETUP_CUSTOM_EVAL.md` - Complete setup guide (recommended)
   - `CUSTOM_EVAL_GUIDE.md` - Detailed reference guide
   - `dataset_example.json` - Example dataset format
   - `dataset_converters.py` - Tools to convert existing datasets

---

## 🚀 Quick Start (3 Steps)

### Step 1: Prepare Your Dataset

Format your data as JSON:

```json
{
  "data": [
    {
      "id": "1",
      "image_path": "/absolute/path/to/image.jpg",
      "question": "Is there an anomaly?",
      "answer": "Yes, there is an anomaly"
    }
  ]
}
```

Use `dataset_converters.py` to convert from CSV, directories, or COCO format.

### Step 2: Update Configuration

Edit `lmms-eval/lmms_eval/tasks/carmel_vad_custom/carmel_vad_custom.yaml`:

```yaml
dataset_path: json
dataset_kwargs:
  data_files: /path/to/your/dataset.json  # ← UPDATE THIS
  split: "train"
```

### Step 3: Run Evaluation

```bash
cd lmms-eval

# Test with 10 samples (1-2 minutes)
bash ../examples/models/carmel_vad_test.sh

# Full evaluation
bash ../examples/models/carmel_vad_custom.sh
```

✅ **Done!** Results in `logs/carmel_vad_eval/`

---

## 📋 File Guide

| File | Purpose | Action |
|------|---------|--------|
| `SETUP_CUSTOM_EVAL.md` | **START HERE** - Step-by-step guide | Read first |
| `CUSTOM_EVAL_GUIDE.md` | Comprehensive reference | Reference |
| `dataset_example.json` | Example dataset format | Copy & modify |
| `dataset_converters.py` | Convert existing data | Use to prepare data |
| `carmel_vad_custom.sh` | Main evaluation | Run after config |
| `carmel_vad_test.sh` | Quick test | Run first to verify |

---

## ⚙️ Key Customizations

### Change Model Path
**File:** `lmms-eval/examples/models/carmel_vad_custom.sh`
```bash
MODEL="/output/sft_qwen3_4b_carmel_vad/pytorch_model.bin"
```

### Change Batch Size (for GPU memory)
**File:** `lmms-eval/examples/models/carmel_vad_custom.sh`
```bash
BATCH_SIZE=16  # Lower if OOM error
```

### Change Number of GPUs
**File:** `lmms-eval/examples/models/carmel_vad_custom.sh`
```bash
TENSOR_PARALLEL_SIZE=4  # Or 1, 2, 8 depending on your setup
```

### Customize Prompts
**File:** `lmms-eval/lmms_eval/tasks/carmel_vad_custom/carmel_vad_custom.yaml`
```yaml
lmms_eval_specific_kwargs:
  default:
    pre_prompt: "You are an anomaly detection expert. "
    post_prompt: "\nAnswer concisely."
```

### Add Custom Metrics
**File:** `lmms-eval/lmms_eval/tasks/carmel_vad_custom/utils.py`
```python
def carmel_vad_aggregate_results_custom(results):
    # Your metric logic
    return {"your_metric": value}
```

Then add to YAML:
```yaml
metric_list:
  - metric: custom_metric
    aggregation: !function utils.carmel_vad_aggregate_results_custom
```

---

## 🔄 Data Format Examples

### Single Image with Classification
```json
{
  "data": [
    {
      "id": "img_001",
      "image_path": "/data/images/img1.jpg",
      "question": "Is there an anomaly?",
      "answer": "Yes",
      "category": "anomaly_detection"
    }
  ]
}
```

### Video Understanding
```json
{
  "data": [
    {
      "id": "vid_001",
      "video_path": "/data/videos/video1.mp4",
      "question": "Describe the activity",
      "answer": "A person walks across the room"
    }
  ]
}
```

### Multi-Image Sequence
```json
{
  "data": [
    {
      "id": "seq_001",
      "image_paths": ["/data/frame1.jpg", "/data/frame2.jpg"],
      "question": "What changes between frames?",
      "answer": "The person moves to the left"
    }
  ]
}
```

---

## 🛠️ Convert Your Existing Data

### From Directory:
```python
from dataset_converters import DirectoryToDataset

DirectoryToDataset.from_directory(
    image_dir="/path/to/images",
    output_path="dataset.json",
    annotation_dir="/path/to/annotations"
)
```

### From CSV:
```python
from dataset_converters import CSVToDataset

CSVToDataset.from_csv(
    csv_path="data.csv",
    output_path="dataset.json",
    image_col="image_path",
    question_col="question",
    answer_col="answer"
)
```

### From COCO format:
```python
from dataset_converters import COCOToDataset

COCOToDataset.from_coco(
    coco_json_path="annotations.json",
    output_path="dataset.json",
    image_dir="/path/to/images"
)
```

---

## 📊 Evaluate Multiple Variants

```bash
# Test different models
for model in /output/model_v1 /output/model_v2 /output/model_v3; do
    sed -i "s|^MODEL=.*|MODEL=$model|" examples/models/carmel_vad_custom.sh
    bash examples/models/carmel_vad_custom.sh
done

# Results will be in logs/carmel_vad_eval/
ls -la logs/carmel_vad_eval/
```

---

## 🐛 Common Issues & Fixes

| Issue | Solution |
|-------|----------|
| "Task not found" | Verify YAML in `lmms_eval/tasks/carmel_vad_custom/` |
| "File not found" | Use absolute paths in YAML file |
| CUDA OOM | Reduce `BATCH_SIZE` or `TENSOR_PARALLEL_SIZE` |
| Slow evaluation | Increase batch size or reduce `max_new_tokens` |
| Empty results | Check dataset JSON format matches example |

---

## 📝 Next Steps

1. **Read:** `SETUP_CUSTOM_EVAL.md` (comprehensive guide)
2. **Prepare:** Convert your data to JSON format
3. **Configure:** Update paths in YAML file
4. **Test:** Run `carmel_vad_test.sh`
5. **Evaluate:** Run `carmel_vad_custom.sh`
6. **Analyze:** Check results in `logs/carmel_vad_eval/`

---

## 📞 Support

For detailed information:
- Setup guide: `SETUP_CUSTOM_EVAL.md`
- Reference: `CUSTOM_EVAL_GUIDE.md`
- Examples: `dataset_example.json`, `dataset_converters.py`

For lmms-eval docs: See `lmms-eval/docs/` or `lmms-eval/README.md`

---

**You're all set!** 🎉

Start with `SETUP_CUSTOM_EVAL.md` for the complete walkthrough.
