# 📖 Carmel VAD Custom Evaluation - Documentation Index

## 🎯 Start Here

**New to this setup?** Read these in order:

1. **[COMPLETE_SUMMARY.md](COMPLETE_SUMMARY.md)** ← **YOU ARE HERE**
2. **[QUICK_REFERENCE.md](QUICK_REFERENCE.md)** - One-page quick start
3. **[SETUP_CUSTOM_EVAL.md](SETUP_CUSTOM_EVAL.md)** - Complete walkthrough

---

## 📚 Documentation Files

### Getting Started
- **[COMPLETE_SUMMARY.md](COMPLETE_SUMMARY.md)** 
  - Overview of what was created
  - 3-step quick start
  - File reference guide
  - ⏱️ 5 minutes to read

- **[QUICK_REFERENCE.md](QUICK_REFERENCE.md)**
  - One-page summary
  - Common customizations
  - Quick troubleshooting table
  - ⏱️ 3 minutes to read

### Detailed Guides
- **[SETUP_CUSTOM_EVAL.md](SETUP_CUSTOM_EVAL.md)**
  - Step-by-step walkthrough
  - Dataset format examples
  - How to customize each component
  - Advanced usage patterns
  - ⏱️ 20 minutes to read fully

- **[CUSTOM_EVAL_GUIDE.md](CUSTOM_EVAL_GUIDE.md)**
  - Technical deep dive
  - API reference for utility functions
  - Custom metric examples
  - Model-specific configurations
  - ⏱️ 30 minutes to read

### System Design & Troubleshooting
- **[ARCHITECTURE_AND_TROUBLESHOOTING.md](ARCHITECTURE_AND_TROUBLESHOOTING.md)**
  - System architecture diagrams
  - Data flow visualization
  - Common errors & solutions
  - Performance optimization tips
  - Debugging techniques
  - ⏱️ 15 minutes to read

---

## 🛠️ Tool & Code Files

### Data Preparation
- **[dataset_example.json](dataset_example.json)**
  - Example dataset structure
  - Shows all supported input formats
  - Use as reference for your data

- **[dataset_converters.py](dataset_converters.py)**
  - Convert from CSV to JSON
  - Convert from directories to JSON
  - Convert from COCO format
  - Convert video datasets
  - **Use:** `python dataset_converters.py` for examples

### Evaluation Scripts
- **[lmms-eval/examples/models/carmel_vad_test.sh](lmms-eval/examples/models/carmel_vad_test.sh)**
  - Quick test with 10 samples (1-2 minutes)
  - Use before full evaluation
  - **Run:** `bash carmel_vad_test.sh`

- **[lmms-eval/examples/models/carmel_vad_custom.sh](lmms-eval/examples/models/carmel_vad_custom.sh)**
  - Full evaluation script
  - Configurable parameters
  - **Run:** `bash carmel_vad_custom.sh`

### Task Implementation
- **[lmms-eval/lmms_eval/tasks/carmel_vad_custom/utils.py](lmms-eval/lmms_eval/tasks/carmel_vad_custom/utils.py)**
  - Data processing functions
  - Metric aggregation
  - **Customize:** Add your own metrics here

- **[lmms-eval/lmms_eval/tasks/carmel_vad_custom/carmel_vad_custom.yaml](lmms-eval/lmms_eval/tasks/carmel_vad_custom/carmel_vad_custom.yaml)**
  - Task configuration
  - Model prompts
  - Metric settings
  - **Update:** Dataset path before running

---

## 🚀 Quick Workflow

### First Time Setup
```
1. Read QUICK_REFERENCE.md (5 min)
   ↓
2. Prepare dataset → dataset_converters.py (5-30 min)
   ↓
3. Update YAML config (2 min)
   ↓
4. Run test script: bash carmel_vad_test.sh (2 min)
   ↓
5. Run full evaluation: bash carmel_vad_custom.sh (varies)
   ↓
6. Check results in logs/carmel_vad_eval/
```

### Running Evaluation
```bash
cd lmms-eval

# Test with small dataset
bash ../examples/models/carmel_vad_test.sh

# Full evaluation
bash ../examples/models/carmel_vad_custom.sh

# Check results
cat logs/carmel_vad_eval/results_*.json | python -m json.tool
```

---

## 📖 Reading Guide by Use Case

### "I just want to run evaluation ASAP"
1. [QUICK_REFERENCE.md](QUICK_REFERENCE.md) - 3 min
2. Prepare dataset
3. Run `carmel_vad_test.sh`
4. Run `carmel_vad_custom.sh`

### "I need help setting everything up"
1. [COMPLETE_SUMMARY.md](COMPLETE_SUMMARY.md) - 5 min
2. [SETUP_CUSTOM_EVAL.md](SETUP_CUSTOM_EVAL.md) - full guide
3. Follow step-by-step instructions

### "I want to customize metrics or prompts"
1. [CUSTOM_EVAL_GUIDE.md](CUSTOM_EVAL_GUIDE.md)
2. Look at examples for your use case
3. Modify `utils.py` or `carmel_vad_custom.yaml`

### "I'm getting errors or issues"
1. [ARCHITECTURE_AND_TROUBLESHOOTING.md](ARCHITECTURE_AND_TROUBLESHOOTING.md)
2. Look up your error in the troubleshooting section
3. Follow the solution

### "I need to convert my existing data"
1. [SETUP_CUSTOM_EVAL.md](SETUP_CUSTOM_EVAL.md) - Step 2
2. Look at examples for your data format
3. Use [dataset_converters.py](dataset_converters.py)

---

## 🎯 Common Tasks

| Task | Documentation | Time |
|------|---------------|------|
| First-time setup | SETUP_CUSTOM_EVAL.md | 20 min |
| Convert CSV data | dataset_converters.py + SETUP_CUSTOM_EVAL.md | 10 min |
| Run evaluation | carmel_vad_custom.sh | varies |
| Fix error | ARCHITECTURE_AND_TROUBLESHOOTING.md | 5-10 min |
| Add custom metric | CUSTOM_EVAL_GUIDE.md | 15 min |
| Change model | carmel_vad_custom.sh | 2 min |
| Use different GPU setup | carmel_vad_custom.sh | 2 min |

---

## 📁 File Structure

```
Carmel_VAD_YQ/
│
├── Documentation (YOU ARE HERE)
│   ├── COMPLETE_SUMMARY.md              ← Overview
│   ├── QUICK_REFERENCE.md               ← Quick start
│   ├── SETUP_CUSTOM_EVAL.md             ← Step-by-step
│   ├── CUSTOM_EVAL_GUIDE.md             ← Technical details
│   └── ARCHITECTURE_AND_TROUBLESHOOTING.md ← System design
│
├── Tools
│   ├── dataset_example.json             ← Example format
│   ├── dataset_converters.py            ← Data conversion
│   └── (other project files)
│
├── lmms-eval/
│   ├── examples/models/
│   │   ├── carmel_vad_test.sh           ← Quick test
│   │   └── carmel_vad_custom.sh         ← Full evaluation
│   │
│   └── lmms_eval/tasks/carmel_vad_custom/
│       ├── utils.py                     ← Processing functions
│       └── carmel_vad_custom.yaml       ← Task config
│
└── logs/
    └── carmel_vad_eval/
        ├── results_*.json               ← Results here
        ├── samples_*.jsonl              ← Predictions
        └── metrics_summary.txt          ← Summary
```

---

## ✨ What You Have

✅ **Complete evaluation framework** - Ready to use  
✅ **Example scripts** - Just update paths and run  
✅ **Data converters** - Convert existing data easily  
✅ **5 documentation guides** - From quick start to deep dive  
✅ **Troubleshooting guide** - Solutions for common issues  
✅ **Example datasets** - See exactly what format needed  

---

## 🚀 Getting Started Now

### Fastest Way (if your data is already in JSON):
```bash
# 1. Update dataset path in YAML
# Edit: lmms-eval/lmms_eval/tasks/carmel_vad_custom/carmel_vad_custom.yaml
# Change: data_files: /path/to/your/dataset.json

# 2. Run test
cd lmms-eval
bash ../examples/models/carmel_vad_test.sh

# 3. Run full evaluation
bash ../examples/models/carmel_vad_custom.sh
```

### If You Need Help:
1. Read [QUICK_REFERENCE.md](QUICK_REFERENCE.md) (3 minutes)
2. Read [SETUP_CUSTOM_EVAL.md](SETUP_CUSTOM_EVAL.md) (20 minutes)
3. Follow step-by-step instructions

---

## 📞 Help & Support

- **Quick answers:** [QUICK_REFERENCE.md](QUICK_REFERENCE.md)
- **Data format:** [SETUP_CUSTOM_EVAL.md](SETUP_CUSTOM_EVAL.md) + [dataset_example.json](dataset_example.json)
- **Errors/Issues:** [ARCHITECTURE_AND_TROUBLESHOOTING.md](ARCHITECTURE_AND_TROUBLESHOOTING.md)
- **Advanced usage:** [CUSTOM_EVAL_GUIDE.md](CUSTOM_EVAL_GUIDE.md)
- **Data conversion:** [dataset_converters.py](dataset_converters.py)

---

## 📝 Note

All documentation is cross-referenced. Links within each doc will take you to related information. Start with the document that matches your current need!

---

**🎉 You're all set! Start with [QUICK_REFERENCE.md](QUICK_REFERENCE.md) or [SETUP_CUSTOM_EVAL.md](SETUP_CUSTOM_EVAL.md).**
