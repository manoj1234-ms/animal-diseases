# Animal Diseases — End-to-end ML project

This repository contains a two-stage classification pipeline for predicting animal disease categories and disease labels.

Quick status

- Models live in `models/` (pre-saved pipelines). Preprocessors produce 26 features; some XGBoost models were saved with different internal `n_features_in_` (26–53). This repository now pads/trims at inference to avoid shape errors, but the recommended long-term fix is to retrain and save pipelines so the saved model sees the exact preprocessor output length.

Setup

1. Create and activate a Python virtual environment (Windows example):

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

2. Install dependencies:

```powershell
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Running inference (local)

- Quick test:

```powershell
python .\test_inference.py
```

- Programmatic usage (example):

```python
from src.inference import predict_disease

sample = { ... }  # see test_inference.py for keys
result = predict_disease(sample)
print(result)
```

What I added

- `src/inference.py`: returns `stage1_confidence` and `stage2_confidence` and handles feature-length mismatches at inference by padding/trimming the transformed feature vector. This is a compatibility layer — prefer retraining.
- `run_multiple_tests.py`: example harness to exercise multiple inputs and show confidences.

Recommended next steps (short-term & long-term)

- Short-term: use the compatibility layer as-is for quick experiments.
- Long-term (recommended): retrain stage-1 and stage-2 pipelines so each saved pipeline's model was trained and saved with the exact preprocessor output. This removes the need for padding/trimming and avoids subtle bugs. Suggested files to add: `scripts/retrain_stage1.py` and `scripts/retrain_stage2.py` to build `Pipeline(preprocessor, model)` and `joblib.dump` them.

Serving

- Option: build a small FastAPI app (place in `serve/app.py`) and run with Uvicorn:

```powershell
uvicorn serve.app:app --reload --host 0.0.0.0 --port 8000
```

Testing & CI

- Add unit tests under `tests/` using `pytest` that cover `predict_disease` and scripts.
- Optionally add GitHub Actions to run tests and lint.

Files to review

- `src/` — model code and training scripts
- `models/` — saved artifacts (inspect using `inspect_models.py`)

If you want, I can now:

- create `scripts/retrain_stage1.py` that builds a consistent pipeline and saves it, or
- add a minimal FastAPI `serve/app.py` and `Dockerfile` to containerize the service.

Pick one next task and I'll implement it.
