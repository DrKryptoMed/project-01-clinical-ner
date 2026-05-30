"""
Loads BC5CDR from HuggingFace, understands its structure,
maps labels to canonical format, and visualises distributions.

Run as script: python notebooks/01_data_exploration.py
"""

from datasets import load_dataset
from collections import Counter
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

sns.set_theme(style="whitegrid", palette="muted")
FIGURES_DIR = Path("outputs/figures")
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# %%
# ── Cell 1: Load the dataset ───────────────────────────────────────────────────
print("Loading BC5CDR from HuggingFace...")
print("First run downloads ~3MB — cached after.\n")

dataset = load_dataset("tner/bc5cdr")

print(dataset)

# %%
# ── Cell 2: Understand the raw structure ──────────────────────────────────────
# Before writing a single loader, read the data as-is.

# What are the features?
print("Features:")
print(dataset["train"].features)
print()

# What does one example look like?
example = dataset["train"][0]
print("Example keys:", list(example.keys()))
print("Tokens:", example["tokens"])
print("Tags  :", example["tags"])

# %%
# ── Cell 3: Decode the label names ────────────────────────────────────────────
# tner/bc5cdr stores tags as raw int32 — no .names attribute.
# We define the mapping ourselves from BC5CDR documentation.

label_names = ["O", "B-Chemical", "I-Chemical", "B-Disease", "I-Disease"]

print("BC5CDR label names (defined from documentation):")
for idx, name in enumerate(label_names):
    print(f"  {idx} → {name}")
# %%
# ── Cell 4: Print a decoded example ───────────────────────────────────────────
# Now decode the first example so you can see the actual annotations.

example = dataset["train"][3]   # pick one with entities
tokens  = example["tokens"]
tag_ids = example["tags"]
tags    = [label_names[t] for t in tag_ids]

print(f"{'Token':<22} {'Tag'}")
print("─" * 38)
for token, tag in zip(tokens, tags):
    marker = " ◄" if tag != "O" else ""
    print(f"{token:<22} {tag}{marker}")

# %%
# ── Cell 5: Split sizes ────────────────────────────────────────────────────────
print("Split sizes:")
for split in ["train", "validation", "test"]:
    n_sentences = len(dataset[split])
    n_tokens    = sum(len(ex["tokens"]) for ex in dataset[split])
    print(f"  {split:<12}: {n_sentences:>5,} sentences  |  {n_tokens:>7,} tokens")

# %%
# ── Cell 6: Label distribution (train) ────────────────────────────────────────
# This tells you the class imbalance — critical for training.

all_tags = [
    label_names[t]
    for ex in dataset["train"]
    for t in ex["tags"]
]

counts = Counter(all_tags)
total  = len(all_tags)

print(f"{'Label':<20} {'Count':>8}  {'%':>7}")
print("─" * 40)
for label in label_names:
    count = counts[label]
    pct   = count / total * 100
    bar   = "█" * int(pct / 2)
    print(f"{label:<20} {count:>8,}  {pct:>6.2f}%  {bar}")

# %%
# ── Cell 7: Sequence length distribution ──────────────────────────────────────
# Tells you how to set max_length in the tokeniser.
# We need to cover most sentences without truncating entities.

seq_lens = [len(ex["tokens"]) for ex in dataset["train"]]

print(f"Sequence length statistics (train):")
print(f"  Min    : {min(seq_lens)}")
print(f"  Max    : {max(seq_lens)}")
print(f"  Mean   : {sum(seq_lens)/len(seq_lens):.1f}")
print(f"  Median : {sorted(seq_lens)[len(seq_lens)//2]}")
print(f"  > 128  : {sum(1 for l in seq_lens if l > 128):,} sentences")
print(f"  > 256  : {sum(1 for l in seq_lens if l > 256):,} sentences")
print(f"  > 512  : {sum(1 for l in seq_lens if l > 512):,} sentences")

# %%
# ── Cell 8: Our canonical label mapping ───────────────────────────────────────
# BC5CDR uses Chemical and Disease.
# We map these to our project's canonical labels.

BC5CDR_TO_CANONICAL = {
    "O"          : "O",
    "B-Chemical" : "B-TREATMENT",
    "I-Chemical" : "I-TREATMENT",
    "B-Disease"  : "B-PROBLEM",
    "I-Disease"  : "I-PROBLEM",
}

print("BC5CDR → Canonical label mapping:")
for source, target in BC5CDR_TO_CANONICAL.items():
    print(f"  {source:<15} → {target}")

print("\nNote: B-TEST and I-TEST are reserved for")
print("Nigerian clinical evaluation notes — not in BC5CDR training data.")

# %%
# ── Cell 9: Distribution chart — save to outputs/figures ──────────────────────
entity_labels  = [l for l in counts if l != "O"]
entity_counts  = [counts[l] for l in entity_labels]
canonical_names = [BC5CDR_TO_CANONICAL[l] for l in entity_labels]
colors = ["#0C447C", "#0C447C", "#A32D2D", "#A32D2D"]

fig, ax = plt.subplots(figsize=(8, 5))
bars = ax.bar(canonical_names, entity_counts,
              color=colors, width=0.5,
              edgecolor="white", linewidth=0.8)

for bar, count in zip(bars, entity_counts):
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 50,
        f"{count:,}",
        ha="center", va="bottom",
        fontsize=12, fontweight="bold"
    )

ax.set_title("Entity Distribution — BC5CDR Training Set",
             fontsize=14, fontweight="bold", pad=16)
ax.set_xlabel("Canonical entity label", fontsize=12)
ax.set_ylabel("Token count", fontsize=12)
sns.despine()
plt.tight_layout()

path = FIGURES_DIR / "bc5cdr_entity_distribution.png"
plt.savefig(path, dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved: {path}")

# %%
# ── Cell 10: Multi-word entity analysis ───────────────────────────────────────
# How many entities span multiple words?
# This informs how well BIO tagging handles our data.

def extract_entities(tokens, tag_ids, label_names):
    """Extracts entity spans from a BIO-tagged sequence."""
    entities = []
    current  = []
    current_type = None

    for token, tid in zip(tokens, tag_ids):
        tag = label_names[tid]
        if tag.startswith("B-"):
            if current:
                entities.append((" ".join(current), current_type))
            current      = [token]
            current_type = tag[2:]
        elif tag.startswith("I-") and current:
            current.append(token)
        else:
            if current:
                entities.append((" ".join(current), current_type))
            current      = []
            current_type = None

    if current:
        entities.append((" ".join(current), current_type))

    return entities

# Collect all entities from train
all_entities = [
    entity
    for ex in dataset["train"]
    for entity in extract_entities(
        ex["tokens"], ex["tags"], label_names
    )
]

df = pd.DataFrame(all_entities, columns=["text", "type"])
df["word_count"] = df["text"].str.split().str.len()
df["canonical"]  = df["type"].map(
    {"Chemical": "TREATMENT", "Disease": "PROBLEM"}
)

print("Entity span length (word count) by type:")
print(df.groupby("canonical")["word_count"]
        .describe()
        .round(2)
        .to_string())

print(f"\nSingle-word entities : {(df.word_count == 1).sum():,}")
print(f"Multi-word entities  : {(df.word_count > 1).sum():,}")

# %%
print("\nExploration complete.")
print("Figures saved to outputs/figures/")
print("Next: formalise the loader in src/data/bc5cdr_loader.py")