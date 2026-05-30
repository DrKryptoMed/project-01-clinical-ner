"""
BC5CDR Data Loader — src/data/bc5cdr_loader.py
================================================
Loads BC5CDR from HuggingFace and converts it to
our canonical label format for training.

BC5CDR entity types and our mapping:
  Chemical → TREATMENT
  Disease  → PROBLEM

Usage:
    from src.data.bc5cdr_loader import BC5CDRLoader
    loader = BC5CDRLoader()
    train, val, test = loader.load()
"""

from dataclasses import dataclass, field
from collections import Counter
from datasets import load_dataset


# ── Label mapping ──────────────────────────────────────────────────────────────
# tner/bc5cdr stores tags as raw int32 — label names defined from documentation

BC5CDR_LABEL_NAMES = [
    "O",
    "B-Chemical", "I-Chemical",
    "B-Disease",  "I-Disease",
]

BC5CDR_TO_CANONICAL = {
    "O"          : "O",
    "B-Chemical" : "B-TREATMENT",
    "I-Chemical" : "I-TREATMENT",
    "B-Disease"  : "B-PROBLEM",
    "I-Disease"  : "I-PROBLEM",
}


# ── Data structure ─────────────────────────────────────────────────────────────

@dataclass
class NERExample:
    """
    A single tokenised sentence with BIO labels.
    Uses canonical label strings — not raw integers.
    """
    example_id : str
    tokens     : list[str]
    labels     : list[str]

    def __post_init__(self):
        assert len(self.tokens) == len(self.labels), (
            f"Token/label mismatch in {self.example_id}: "
            f"{len(self.tokens)} tokens vs {len(self.labels)} labels"
        )

    @property
    def entity_spans(self) -> list[tuple[str, str]]:
        """Returns (entity_text, entity_type) for each entity."""
        spans   = []
        current = []
        c_type  = None

        for token, label in zip(self.tokens, self.labels):
            if label.startswith("B-"):
                if current:
                    spans.append((" ".join(current), c_type))
                current = [token]
                c_type  = label[2:]
            elif label.startswith("I-") and current:
                current.append(token)
            else:
                if current:
                    spans.append((" ".join(current), c_type))
                current = []
                c_type  = None

        if current:
            spans.append((" ".join(current), c_type))

        return spans


# ── Loader class ───────────────────────────────────────────────────────────────

class BC5CDRLoader:
    """
    Loads BC5CDR from HuggingFace tner/bc5cdr and converts
    all labels to our canonical format.
    """

    HF_DATASET = "tner/bc5cdr"

    def __init__(self, cache_dir: str = "./outputs/hf_cache"):
        self.cache_dir    = cache_dir
        self._dataset     = None

    def _fetch(self) -> None:
        """Downloads and caches the dataset if not already loaded."""
        if self._dataset is None:
            self._dataset = load_dataset(
                self.HF_DATASET,
                cache_dir=self.cache_dir
            )

    def _convert_example(
        self,
        example : dict,
        idx     : int,
        split   : str
    ) -> NERExample:
        """Converts one raw HuggingFace example to NERExample."""
        tokens = example["tokens"]
        labels = [
            BC5CDR_TO_CANONICAL.get(
                BC5CDR_LABEL_NAMES[tid], "O"
            )
            for tid in example["tags"]
        ]
        return NERExample(
            example_id = f"{split}_{idx}",
            tokens     = tokens,
            labels     = labels
        )

    def load(self) -> tuple[
        list[NERExample],
        list[NERExample],
        list[NERExample]
    ]:
        """
        Loads all three splits.
        Returns: train, validation, test
        """
        self._fetch()

        result = {}
        for split in ["train", "validation", "test"]:
            result[split] = [
                self._convert_example(ex, idx, split)
                for idx, ex in enumerate(self._dataset[split])
            ]

        return result["train"], result["validation"], result["test"]

    def get_stats(self, examples: list[NERExample]) -> dict:
        """Summary statistics for a split."""
        all_labels = [l for ex in examples for l in ex.labels]
        all_spans  = [s for ex in examples for s in ex.entity_spans]
        type_counts = Counter(t for _, t in all_spans)

        return {
            "num_examples"             : len(examples),
            "total_tokens"             : len(all_labels),
            "total_entities"           : len(all_spans),
            "entities_by_type"         : dict(type_counts),
            "avg_tokens_per_example"   : round(
                len(all_labels) / len(examples), 1
            ),
            "avg_entities_per_example" : round(
                len(all_spans) / len(examples), 2
            ),
        }