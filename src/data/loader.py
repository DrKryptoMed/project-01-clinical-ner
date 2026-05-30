"""
i2b2 Data Loader — src/data/loader.py
======================================
Loads clinical notes and concept annotations
from i2b2 2010 format (.txt and .con files).

Usage:
    from src.data.loader import I2B2DataLoader
    loader = I2B2DataLoader("data/sample")
    notes  = loader.load_all()
"""

import re
from dataclasses import dataclass, field
from pathlib import Path


# ── Data structures ────────────────────────────────────────────────────────────

@dataclass
class Concept:
    """A single annotated clinical entity."""
    text       : str
    start_line : int
    start_word : int
    end_line   : int
    end_word   : int
    label      : str

    @property
    def label_upper(self) -> str:
        return self.label.upper()


@dataclass
class ClinicalNote:
    """A single clinical note with its text and annotated concepts."""
    note_id  : str
    lines    : list[str]
    concepts : list[Concept] = field(default_factory=list)

    @property
    def full_text(self) -> str:
        return "\n".join(self.lines)

    @property
    def word_count(self) -> int:
        return sum(len(line.split()) for line in self.lines)

    @property
    def concept_count(self) -> int:
        return len(self.concepts)

    def concepts_by_type(self, label: str) -> list[Concept]:
        return [c for c in self.concepts if c.label == label.lower()]


# ── Parser functions ────────────────────────────────────────────────────────────

def parse_txt_file(path: Path) -> list[str]:
    with open(path, "r", encoding="utf-8") as f:
        return [line.rstrip("\n") for line in f.readlines()]


def parse_con_file(path: Path) -> list[Concept]:
    pattern = re.compile(
        r'c="(?P<text>[^"]+)"\s+'
        r'(?P<sl>\d+):(?P<sw>\d+)\s+'
        r'(?P<el>\d+):(?P<ew>\d+)'
        r'\|\|t="(?P<label>[^"]+)"'
    )
    concepts = []
    with open(path, "r", encoding="utf-8") as f:
        for line_num, raw_line in enumerate(f, start=1):
            raw_line = raw_line.strip()
            if not raw_line:
                continue
            match = pattern.match(raw_line)
            if not match:
                print(f"  [WARNING] Could not parse line {line_num} "
                      f"in {path.name}: {raw_line!r}")
                continue
            concepts.append(Concept(
                text       = match.group("text"),
                start_line = int(match.group("sl")),
                start_word = int(match.group("sw")),
                end_line   = int(match.group("el")),
                end_word   = int(match.group("ew")),
                label      = match.group("label"),
            ))
    return concepts


# ── Main loader class ──────────────────────────────────────────────────────────

class I2B2DataLoader:
    """Loads i2b2 2010 clinical notes from a directory."""

    VALID_LABELS = {"problem", "treatment", "test"}

    def __init__(self, data_dir: str | Path):
        self.data_dir = Path(data_dir)
        if not self.data_dir.exists():
            raise FileNotFoundError(
                f"Data directory not found: {self.data_dir}\n"
                f"Run: python scripts/create_sample_data.py"
            )

    def load_note(self, note_id: str) -> ClinicalNote:
        txt_path = self.data_dir / f"{note_id}.txt"
        con_path = self.data_dir / f"{note_id}.con"
        if not txt_path.exists():
            raise FileNotFoundError(f"Text file not found: {txt_path}")
        lines    = parse_txt_file(txt_path)
        concepts = parse_con_file(con_path) if con_path.exists() else []
        for concept in concepts:
            if concept.label not in self.VALID_LABELS:
                print(f"  [WARNING] Unknown label '{concept.label}' in {note_id}")
        return ClinicalNote(note_id=note_id, lines=lines, concepts=concepts)

    def load_all(self) -> list[ClinicalNote]:
        txt_files = sorted(self.data_dir.glob("*.txt"))
        if not txt_files:
            raise FileNotFoundError(
                f"No .txt files found in {self.data_dir}"
            )
        return [self.load_note(f.stem) for f in txt_files]

    def get_stats(self, notes: list[ClinicalNote]) -> dict:
        all_concepts = [c for n in notes for c in n.concepts]
        return {
            "num_notes"             : len(notes),
            "total_concepts"        : len(all_concepts),
            "by_type": {
                "PROBLEM"   : sum(1 for c in all_concepts
                                  if c.label == "problem"),
                "TREATMENT" : sum(1 for c in all_concepts
                                  if c.label == "treatment"),
                "TEST"      : sum(1 for c in all_concepts
                                  if c.label == "test"),
            },
            "avg_concepts_per_note" : round(
                len(all_concepts) / len(notes), 2
            ) if notes else 0,
        }