# 📊 Carmel VAD Custom Evaluation - Complete Setup Summary

## 🎯 What You Got

I've created a **complete, production-ready framework** for evaluating your fine-tuned Carmel VAD model on custom datasets using lmms-eval. Here's what was set up:

### ✨ New Components Created

#### 1. **Custom Task Implementation** 
```
lmms-eval/lmms_eval/tasks/carmel_vad_custom/
├── carmel_vad_custom.yaml      (Task configuration)
└── utils.py                    (Data processing functions)
```

**What it does:**
- Loads your custom dataset from JSON
- Processes images/videos and prompts
- Runs model inference via vLLM
- Computes evaluation metrics (accuracy, precision, recall, F1)
- Generates detailed result reports

#### 2. **Evaluation Scripts**
```
lmms-eval/examples/models/
├── carmel_vad_custom.sh        (Full evaluation - main script)
├── carmel_vad_test.sh          (Quick test with 10 samples)
└── carmel_vad_qwen3vl.sh       (Original template)
```

**What they do:**
- `carmel_vad_test.sh` - Quick sanity check (1-2 min)
- `carmel_vad_custom.sh` - Full evaluation on entire dataset

#### 3. **Documentation** (5 guides)
```
Project Root/
├── QUICK_REFERENCE.md                          (Start here!)
├── SETUP_CUSTOM_EVAL.md                        (Step-by-step guide)
├── CUSTOM_EVAL_GUIDE.md                        (Detailed reference)
├── ARCHITECTURE_AND_TROUBLESHOOTING.md         (System design & fixes)
└── (This file - Complete Summary)
```

#### 4. **Data Tools**
```
Project Root/
├── dataset_example.json                        (Example dataset format)
└── dataset_converters.py                       (Convert CSV/directories/COCO)
```

---

## 🚀 Quick Start (Follow These 3 Steps)

### **Step 1: Prepare Your Data** (5 minutes)

Convert your data to JSON format:

```json
{
  "data": [
    {
      "id": "sample_001",
      "image_path": "/absolute/path/to/image.jpg",
      "question": "Is there an anomaly?",
      "answer": "Yes, there is"
    }
  ]
}
```

**Need to convert from another format?**
```python
from dataset_converters import *

# From CSV
CSVToDataset.from_csv("data.csv", "dataset.json", 
                      image_col="img", question_col="q", answer_col="ans")

# From directory
DirectoryToDataset.from_directory("/path/to/images", "dataset.json")

# From COCO
COCOToDataset.from_coco("annotations.json", "dataset.json", "/path/to/images")

# From video directory  
VideoDatasetConverter.from_video_directory("/videos", "video_dataset.json")
```

### **Step 2: Configure Evaluation** (2 minutes)

Edit `lmms-eval/lmms_eval/tasks/carmel_vad_custom/carmel_vad_custom.yaml`:

```yaml
dataset_path: json
dataset_kwargs:
  data_files: /path/to/your/dataset.json    # ← CHANGE THIS
  split: "train"
```

That's it! Everything else is pre-configured.

### **Step 3: Run Evaluation** (varies by dataset size)

```bash
cd lmms-eval

# Test first (10 samples, ~1-2 minutes)
bash ../examples/models/carmel_vad_test.sh

# Full evaluation (depends on your dataset size)
bash ../examples/models/carmel_vad_custom.sh
```

✅ **Done!** Results in `logs/carmel_vad_eval/`

---

## 📁 File Reference Guide

| File | Purpose | When to Use |
|------|---------|------------|
| **QUICK_REFERENCE.md** | One-page summary of everything | First read! |
| **SETUP_CUSTOM_EVAL.md** | Complete step-by-step walkthrough | Detailed setup help |
| **CUSTOM_EVAL_GUIDE.md** | Technical reference with examples | Implementation details |
| **ARCHITECTURE_AND_TROUBLESHOOTING.md** | System design + debugging guide | Troubleshooting issues |
| **dataset_example.json** | Example dataset structure | Reference format |
| **dataset_converters.py** | Convert existing data to JSON | Data preparation |
| **carmel_vad_test.sh** | Quick test script | Initial verification |
| **carmel_vad_custom.sh** | Full evaluation script | Main evaluation |
| **carmel_vad_custom.yaml** | Task configuration | Customize for your task |
| **utils.py** | Data processing functions | Advanced customization |

---

## 🔧 Customization Quick Reference

### Change Model
Edit `lmms-eval/examples/models/carmel_vad_custom.sh`:
```bash
MODEL="/path/to/your/fine-tuned/model"
```

### Adjust Batch Size (for GPU memory)
```bash
BATCH_SIZE=8  # Lower if OOM, increase if GPU not saturated
```

### Use Different Number of GPUs
```bash
TENSOR_PARALLEL_SIZE=2  # Use 2 GPUs instead of 4
```

### Customize Prompts
Edit `carmel_vad_custom.yaml`:
```yaml
lmms_eval_specific_kwargs:
  default:
    pre_prompt: "You are an anomaly detection expert. "
    post_prompt: "\nProvide a one-sentence answer."
```

### Add Custom Metrics
1. Add function to `utils.py`:
```python
def custom_metric(results):
    # Your metric logic
    return {"metric_name": value}
```

2. Add to `carmel_vad_custom.yaml`:
```yaml
metric_list:
  - metric: custom_metric
    aggregation: !function utils.custom_metric
```

### Support Different Input Types
All automatically supported:
- ✅ Single images: `image_path`
- ✅ Videos: `video_path`
- ✅ Multiple images: `image_paths`
- ✅ Pre-loaded images: `image`

---

## 📊 Results Interpretation

After running evaluation, check `logs/carmel_vad_eval/`:

```
results_<timestamp>.json          Main results
├── results
│   └── carmel_vad_custom
│       ├── accuracy: 0.85       (Percentage correct)
│       ├── correct: 425         (Number correct)
│       └── total: 500           (Total samples)
├── configs
│   ├── model: "..."
│   └── task: "carmel_vad_custom"
└── timestamp: "2025-01-05..."

samples_<timestamp>.jsonl         Individual predictions (if --log_samples)
├── {"id": "1", "pred": "Yes", "answer": "Yes", "score": 1.0}
├── {"id": "2", "pred": "No", "answer": "Maybe", "score": 0.5}
└── ...
```

---

## 🆘 Quick Troubleshooting

| Problem | Solution |
|---------|----------|
| "Task not found" | Verify `lmms_eval/tasks/carmel_vad_custom/` exists |
| "File not found" | Use absolute paths in YAML |
| CUDA OOM | Lower `BATCH_SIZE` or reduce `max_new_tokens` |
| Slow evaluation | Increase batch size or reduce token generation |
| No output | Check dataset JSON format matches example |
| Hanging | Set `export NCCL_BLOCKING_WAIT=1` |

See `ARCHITECTURE_AND_TROUBLESHOOTING.md` for detailed troubleshooting.

---

## 📈 Advanced Usage Examples

### Evaluate Multiple Models
```bash
for model in /models/v1 /models/v2 /models/v3; do
    sed -i "s|^MODEL=.*|MODEL=$model|" examples/models/carmel_vad_custom.sh
    bash examples/models/carmel_vad_custom.sh
done
```

### Evaluate on Multiple Dataset Splits
```bash
# Create separate config files
cp carmel_vad_custom.yaml carmel_vad_test_variant.yaml
# Edit to change split: "test"

# Run both
python -m lmms_eval --tasks carmel_vad_custom carmel_vad_test_variant --output_path logs/all_splits
```

### Generate Predictions Without Metrics
Add to `carmel_vad_custom.sh`:
```bash
--output_path ./predictions \
--log_samples \
--log_samples_suffix predictions_only
```

### Use Different Model Backend
Instead of vLLM, use SGLang (faster for Qwen):
```bash
--model sglang \
--model_args model=/path/to/model,dtype=bfloat16
```

---

## 🎓 Understanding the Framework

### Data Processing Pipeline
```
dataset.json
    ↓
doc_to_visual() → Load image/video
    ↓
doc_to_text() → Create prompt
    ↓
Model Inference → Generate response
    ↓
process_results() → Compare with ground truth
    ↓
aggregate_results() → Compute metrics
    ↓
results_*.json
```

### Key Functions in utils.py

| Function | Purpose |
|----------|---------|
| `carmel_vad_doc_to_visual()` | Extract image/video from document |
| `carmel_vad_doc_to_text()` | Create text prompt from document |
| `carmel_vad_process_results()` | Process single model output |
| `carmel_vad_aggregate_results()` | Combine results into metrics |
| `carmel_vad_aggregate_results_classification()` | Multi-class metrics |
| `carmel_vad_aggregate_results_anomaly()` | Binary classification metrics |

---

## 📋 Verification Checklist

Before running full evaluation:

- [ ] Dataset JSON created and validated
- [ ] All image/video paths are absolute
- [ ] YAML `dataset_path` is correct
- [ ] Model path exists
- [ ] GPU memory sufficient for batch size
- [ ] vLLM installed: `pip install vllm>=0.11.0`
- [ ] Test run successful: `bash carmel_vad_test.sh`
- [ ] Results directory is writable

---

## 🔗 Integration Points

This framework integrates with:

1. **Your Model** - Any Qwen3-VL or similar model via vLLM
2. **Your Data** - JSON format (easily converted)
3. **Your Metrics** - Customizable in `utils.py`
4. **Your Infrastructure** - Works with 1+ GPUs

---

## 📚 Documentation Roadmap

```
Start Here
    ↓
QUICK_REFERENCE.md          (1-page overview)
    ↓
SETUP_CUSTOM_EVAL.md        (Step-by-step setup)
    ├→ Need data help?       → dataset_converters.py
    ├→ Need to customize?    → CUSTOM_EVAL_GUIDE.md
    └→ Having issues?        → ARCHITECTURE_AND_TROUBLESHOOTING.md
```

---

## 🎯 Next Actions

1. **Read:** `QUICK_REFERENCE.md` (5 min)
2. **Prepare:** Convert your data to JSON (10-30 min)
3. **Configure:** Update paths in YAML (2 min)
4. **Test:** Run `carmel_vad_test.sh` (2-5 min)
5. **Evaluate:** Run `carmel_vad_custom.sh` (varies)
6. **Analyze:** Check `logs/carmel_vad_eval/` results (5 min)

---

## 💡 Pro Tips

1. **Always test first** - Run `carmel_vad_test.sh` with `--limit 10` before full eval
2. **Monitor GPU** - Open second terminal: `watch nvidia-smi`
3. **Save results** - Copy `logs/carmel_vad_eval/` before next run
4. **Start small** - Test with 100 samples before full dataset
5. **Log samples** - Use `--log_samples` to debug predictions

---

## 📞 Support Resources

- **Setup issues** → `SETUP_CUSTOM_EVAL.md`
- **Data conversion** → `dataset_converters.py`
- **Troubleshooting** → `ARCHITECTURE_AND_TROUBLESHOOTING.md`
- **Advanced usage** → `CUSTOM_EVAL_GUIDE.md`
- **lmms-eval docs** → `lmms-eval/docs/` or `lmms-eval/README.md`

---

## ✅ You're Ready!

Everything is set up and ready to go. 

**Start with:** `SETUP_CUSTOM_EVAL.md` or `QUICK_REFERENCE.md`

**Then run:** `bash lmms-eval/examples/models/carmel_vad_test.sh`

Good luck! 🚀
