# Clinical NLP: EHR Diagnosis Extraction

Part of my **Healthcare AI Portfolio** — 20 production-grade clinical AI projects
built with clinical ethics compliance, explainability, and full deployment.

## Clinical problem
Over 80% of actionable clinical data lives in unstructured EHR notes.
This system extracts diagnoses, medications, lab tests, and procedures
from free-text clinical notes and returns structured FHIR R4 resources.

## Technical approach
- **Model**: BioBERT fine-tuned for token classification (NER)
- **Task**: BIO tagging → entity span extraction
- **Output**: JSON entities + FHIR R4 bundle via FastAPI
- **Explainability**: LIME token attribution
- **Tracking**: MLflow

## Project structure
src/         → modular Python source code
notebooks/   → exploratory data analysis
configs/     → hyperparameter configs
data/        → (not committed) raw and processed data
outputs/     → metrics, figures, model card

## Setup
```bash
python -m venv venv
source venv/bin/activate   # or venv\Scripts\activate on Windows
pip install -r requirements.txt
```

## Ethics & compliance
- WHO Ethics Guidelines for AI in Health
- EU AI Act — High Risk classification acknowledged
- Human-in-the-loop: all extractions require clinician confirmation
- HIPAA: trained on de-identified MIMIC-III (PhysioNet credentialed access)

## Status
🟡 In progress — Session 2: environment setup complete