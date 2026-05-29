"""
Environment verification test for Clinical NER
Run with pytest tests/test_environment.py -v

Verifies that all dependencies are installed correctly and
that BioBERT tokenizer loads from HuggingFace hub
"""

import sys

# Check Python version
def test_python_version():
    assert sys.version_info.major == 3
    assert sys.version_info.minor == 13
    print(f"\nPython version: {sys.version}")
    
# check core ML
def test_torch_import():
    import torch
    assert torch.__version__.startswith("2.6")
    print(f"\nPyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    
def test_transformers_import():
    import transformers
    assert transformers.__version__.startswith("4.47")
    print(f"\nTransformers version: {transformers.__version__}")
    
def test_datasets_import():
    import datasets
    print(f"\nDatasets version: {datasets.__version__}")
    
# BioBERT tokenizer test
def test_biobert_tokenizer_loads():
    from transformers import AutoTokenizer
    
    tokenizer = AutoTokenizer.from_pretrained(
        "dmis-lab/biobert-base-cased-v1.2",
        cache_dir="./outputs/hf_cache"
    )
    # Tokenise a real clinical note fragment
    clinical_text = "Patient with T2DM and HTN started on furosemide 40mg"
    tokens = tokenizer(clinical_text)
    token_strings = tokenizer.convert_ids_to_tokens(tokens['input_ids'])
    
    assert len(tokens["input_ids"]) > 0
    assert "[CLS]" in token_strings
    assert "[SEP]" in token_strings
    
    print(f"\nInput text: {clinical_text}")
    print(f"Tokens: {token_strings}")
    print(f"Token IDs: {tokens['input_ids']}")
    print(f"Token count: {len(tokens['input_ids'])}")
    
    
# Data & evaluation libraries
def test_numpy_import():
    import numpy as np
    assert int(np.__version__.split(".")[0]) >= 2
    print(f"\nNumPy version: {np.__version__}")
    
def test_pandas_import():
    import pandas as pd
    print(f"\nPandas version: {pd.__version__}")
    
def test_sklearn_import():
    import sklearn
    print(f"\nScikit-learn version: {sklearn.__version__}")
    
def test_seqeval_import():
    from seqeval.metrics import f1_score, classification_report
    
    y_true = [["O", "B-PROBLEM", "I-PROBLEM", "O", "B-TREATMENT"]]
    y_pred = [["O", "B-PROBLEM", "I-PROBLEM", "O", "B-TREATMENT"]]
    
    score = f1_score(y_true, y_pred)
    assert score == 1.0
    print(f"\nSeqeval F1 (perfect prediction): {score}")
    print(classification_report(y_true, y_pred))
    
#FHIR output literacy
def test_fhir_import():
    from fhir.resources.condition import Condition
    from fhir.resources.codeableconcept import CodeableConcept
    from fhir.resources.coding import Coding
    
    condition = Condition.construct(
        subject={"reference": "Patient/example"},
        code=CodeableConcept.construct(
            coding=[Coding.construct(
                system="http://snomed.info/sct",
                code="44054006",
                display="Type 2 diabetes mellitus"
            )]
        )
    )
    assert condition is not None
    print(f"\nFHIR Condition constructed: {condition.code.coding[0].display}")
    
# API framework
def test_fastapi_import():
    import fastapi
    print(f"\nFastAPI version: {fastapi.__version__}")
    
# Experiment tracking
def test_mlflow_import():
    import mlflow
    print(f"\nMLflow version: {mlflow.__version__}")
    
# Explainability
def test_lime_import():
    import lime
    import lime.lime_text
    print(f"\nLime imported successfully")