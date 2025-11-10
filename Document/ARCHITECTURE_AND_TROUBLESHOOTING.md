# Carmel VAD Evaluation - System Architecture & Troubleshooting

## System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    Your Fine-Tuned Model                        │
│  /output/sft_qwen3_4b_carmel_vad/pytorch_model.bin              │
└─────────────────┬───────────────────────────────────────────────┘
                  │
                  │ vLLM Backend
                  ▼
┌─────────────────────────────────────────────────────────────────┐
│                  lmms_eval Framework                            │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Task: carmel_vad_custom                                       │
│  ├── carmel_vad_custom.yaml  ← Configuration                    │
│  ├── utils.py               ← Processing functions              │
│  └── Dataset.json           ← Your data                         │
│                                                                 │
│  Process Flow:                                                 │
│  1. Load dataset.json                                          │
│  2. For each sample:                                           │
│     - Extract image/video (doc_to_visual)                      │
│     - Create prompt (doc_to_text)                              │
│     - Run model inference                                      │
│     - Process results (process_results)                        │
│  3. Aggregate results (aggregate_results)                      │
│  4. Generate metrics                                           │
│                                                                 │
└─────────────────┬───────────────────────────────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────────────────────────────┐
│                       Output Files                              │
├─────────────────────────────────────────────────────────────────┤
│  logs/carmel_vad_eval/                                          │
│  ├── results_<timestamp>.json    ← Main results                │
│  ├── samples_<timestamp>.jsonl   ← Individual predictions       │
│  └── metrics_summary.txt         ← Formatted summary            │
└─────────────────────────────────────────────────────────────────┘
```

## Data Flow Diagram

```
Your Dataset (JSON)
│
├─ Sample 1:
│  ├─ id: "sample_001"
│  ├─ image_path: "/data/img1.jpg"
│  ├─ question: "Is there an anomaly?"
│  └─ answer: "Yes, there is"
│
└─ Sample 2:
   ├─ id: "sample_002"
   ├─ video_path: "/data/video1.mp4"
   ├─ question: "Describe activity"
   └─ answer: "Person walks"
        │
        ▼ (for each sample)
   ┌─────────────────────────────┐
   │ carmel_vad_doc_to_visual    │  Load image/video
   ├─────────────────────────────┤
   │ Returns: [PIL Image/video]  │
   └──────────────┬──────────────┘
                  │
                  ▼ (with question)
   ┌─────────────────────────────┐
   │ carmel_vad_doc_to_text      │  Create prompt
   ├─────────────────────────────┤
   │ Returns: "You are... Q..."  │
   └──────────────┬──────────────┘
                  │
                  ▼ (model inference)
   ┌─────────────────────────────┐
   │ Model Inference             │  Generate response
   ├─────────────────────────────┤
   │ Returns: "Model output"     │
   └──────────────┬──────────────┘
                  │
                  ▼ (process results)
   ┌─────────────────────────────┐
   │ carmel_vad_process_results  │  Compare pred vs ground truth
   ├─────────────────────────────┤
   │ Returns: {metric_data}      │
   └──────────────┬──────────────┘
                  │
                  ▼ (aggregate all)
   ┌─────────────────────────────┐
   │ carmel_vad_aggregate_...    │  Compute final metrics
   ├─────────────────────────────┤
   │ Returns: {accuracy: 0.85}   │
   └─────────────────────────────┘
```

## Configuration File Structure

```
carmel_vad_custom.yaml
├── Dataset Section
│   ├── dataset_path: json
│   └── dataset_kwargs:
│       ├── data_files: /path/to/dataset.json
│       └── split: train
│
├── Task Definition
│   ├── task: carmel_vad_custom
│   ├── test_split: test
│   └── output_type: generate_until
│
├── Document Processing
│   ├── doc_to_visual: !function utils.carmel_vad_doc_to_visual
│   ├── doc_to_text: !function utils.carmel_vad_doc_to_text
│   └── doc_to_target: answer
│
├── Generation Config
│   ├── generation_kwargs:
│   │   ├── max_new_tokens: 256
│   │   ├── temperature: 0.7
│   │   ├── top_p: 0.9
│   │   └── do_sample: true
│   │
├── Metrics
│   └── metric_list:
│       └── carmel_vad_score:
│           ├── aggregation: utils.carmel_vad_aggregate_results
│           └── higher_is_better: true
│
└── Model-Specific Config
    └── lmms_eval_specific_kwargs:
        └── default:
            ├── pre_prompt: ""
            └── post_prompt: "\nAnswer."
```

## Directory Structure

```
Carmel_VAD_YQ/
│
├── lmms-eval/                          # lmms-eval framework
│   ├── lmms_eval/
│   │   ├── models/                     # Model implementations
│   │   │   └── chat/
│   │   │       ├── vllm.py             # vLLM backend (used)
│   │   │       └── ...
│   │   │
│   │   └── tasks/                      # Benchmark tasks
│   │       └── carmel_vad_custom/      # ✨ YOUR CUSTOM TASK
│   │           ├── utils.py            # Processing functions
│   │           └── carmel_vad_custom.yaml
│   │
│   └── examples/
│       └── models/
│           ├── carmel_vad_custom.sh    # ✨ Main evaluation
│           └── carmel_vad_test.sh      # ✨ Quick test
│
├── QUICK_REFERENCE.md                  # Quick start guide
├── SETUP_CUSTOM_EVAL.md                # Step-by-step setup
├── CUSTOM_EVAL_GUIDE.md                # Detailed reference
├── dataset_example.json                # Example dataset
├── dataset_converters.py               # Data conversion tools
│
├── your_dataset.json                   # YOUR DATASET ← PUT HERE
│
└── logs/                               # Output directory
    └── carmel_vad_eval/
        ├── results_*.json              # Final results
        ├── samples_*.jsonl             # Individual predictions
        └── metrics_summary.txt         # Summary
```

## Troubleshooting Flowchart

```
Does evaluation run?
├─ NO ──┬─ YAML file error?
│       │  └─ Check: Task name matches YAML filename
│       │  └─ Check: All YAML syntax is valid (use yamllint)
│       │
│       ├─ Dataset loading error?
│       │  └─ Check: Dataset path is absolute
│       │  └─ Check: JSON file exists and is valid
│       │  └─ Check: JSON has "data" key with list
│       │
│       ├─ Import error?
│       │  └─ Solution: pip install -e . (from lmms-eval dir)
│       │
│       └─ GPU error?
│           └─ Check: nvidia-smi shows GPUs
│           └─ Solution: Reduce TENSOR_PARALLEL_SIZE
│
└─ YES ──┬─ OOM Error?
         │  └─ Solution: Lower BATCH_SIZE
         │  └─ Solution: Lower GPU_MEMORY_UTILIZATION
         │  └─ Solution: Reduce max_new_tokens in YAML
         │
         ├─ Slow evaluation?
         │  └─ Increase BATCH_SIZE if memory allows
         │  └─ Reduce max_new_tokens if possible
         │  └─ Use more GPUs (increase TENSOR_PARALLEL_SIZE)
         │
         ├─ Hanging/Stuck?
         │  └─ Solution: Set NCCL_BLOCKING_WAIT=1
         │  └─ Solution: Increase NCCL_TIMEOUT
         │
         └─ Wrong results?
             └─ Check: Dataset JSON format matches example
             └─ Check: Image/video paths are correct
             └─ Check: Prompts in YAML match your task
             └─ Check: Metric function is correct
```

## Common Error Messages & Solutions

### Error: "Module not found: lmms_eval"
```
Cause: Not installed in development mode
Solution:
  cd lmms-eval
  pip install -e .
```

### Error: "Task 'carmel_vad_custom' not found"
```
Cause: YAML file not found or wrong location
Solution:
  ls -la lmms_eval/tasks/carmel_vad_custom/carmel_vad_custom.yaml
  # Should exist and match task name
```

### Error: "No such file: /path/to/dataset.json"
```
Cause: Using relative path instead of absolute
Solution:
  # Use absolute paths in YAML:
  data_files: /home/yuqiang/yl4300/project/.../dataset.json
  # Verify: ls -la /path/to/dataset.json
```

### Error: "CUDA out of memory"
```
Cause: Batch size or model size too large
Solutions (in order):
  1. Reduce BATCH_SIZE: 16 → 8 or 4
  2. Reduce GPU_MEMORY_UTILIZATION: 0.85 → 0.7 or 0.5
  3. Reduce max_new_tokens: 256 → 128
  4. Use fewer GPUs: TENSOR_PARALLEL_SIZE=2
```

### Error: "JSON decode error"
```
Cause: Invalid dataset JSON format
Solution:
  python -m json.tool your_dataset.json
  # If error, check format matches example
```

### Error: "Image loading failed"
```
Cause: Image path incorrect or file corrupted
Solution:
  # Verify paths in JSON:
  python -c "import json; d=json.load(open('dataset.json')); print(d['data'][0]['image_path'])"
  # Check file exists:
  file /path/to/image.jpg
```

### Hanging after "Model Responding"
```
Cause: GPU synchronization issue
Solution:
  export NCCL_BLOCKING_WAIT=1
  export NCCL_TIMEOUT=36000000
  bash examples/models/carmel_vad_custom.sh
```

## Performance Optimization

### For Fast Evaluation:
```bash
# Reduce token generation
generation_kwargs:
  max_new_tokens: 64  # Smaller = faster

# Increase batch size (if GPU memory allows)
BATCH_SIZE=32

# Use more GPUs
TENSOR_PARALLEL_SIZE=8
```

### For Highest Quality:
```bash
# Allow longer generation
generation_kwargs:
  max_new_tokens: 512
  temperature: 0.7    # More randomness for diversity

# Use single GPU for consistent inference
TENSOR_PARALLEL_SIZE=1
BATCH_SIZE=4
```

### For Limited GPU Memory:
```bash
# Minimum resources
BATCH_SIZE=2
GPU_MEMORY_UTILIZATION=0.4
TENSOR_PARALLEL_SIZE=1

# In YAML:
generation_kwargs:
  max_new_tokens: 128
```

## Monitoring and Profiling

### Monitor GPU Usage:
```bash
# In another terminal
watch nvidia-smi

# Or with more detail
nvidia-smi --query-gpu=index,name,utilization.gpu,utilization.memory,memory.used,memory.total --format=csv -l 1
```

### Profile Evaluation:
```bash
# First, run with small dataset
--limit 100

# Check timing
cat logs/carmel_vad_eval/results_*.json | python -c "
import sys, json
data = json.load(sys.stdin)
print('Total time:', data.get('walltime'))
"
```

## Debugging Tips

### Enable Verbose Logging:
In script, add:
```bash
export CUDA_LAUNCH_BLOCKING=1
export NCCL_DEBUG=INFO
```

### Check Individual Samples:
```bash
# View first few samples
python -c "
import json
with open('logs/carmel_vad_eval/samples_*.jsonl') as f:
    for i, line in enumerate(f):
        if i < 5: print(json.loads(line))
"
```

### Validate Dataset:
```python
import json
with open('dataset.json') as f:
    data = json.load(f)
    print(f'Total samples: {len(data[\"data\"])}')
    for sample in data['data'][:3]:
        print(sample)
        print('---')
```

---

## Visual Checklist

Before running evaluation:

```
□ Dataset JSON created and validated
□ All image/video paths are absolute
□ YAML dataset_path updated correctly
□ Model path is correct
□ GPU memory sufficient for batch size
□ vLLM installed: pip install vllm>=0.11.0
□ lmms-eval installed: pip install -e .
□ Test run successful: bash carmel_vad_test.sh
□ Output directory writable
```

After evaluation:

```
□ Check results_*.json exists
□ Review metrics in file
□ Check samples_*.jsonl if needed
□ Verify accuracy/metrics make sense
□ Save results to permanent location
```

---

For more details, see `SETUP_CUSTOM_EVAL.md`
