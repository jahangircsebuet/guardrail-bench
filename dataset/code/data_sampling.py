# ============================================================
# FULL FILTERING PIPELINE (BERTScore + COMET)
#
# Outputs:
#   - merged_scores.csv
#   - sampled_gold.csv
#   - sampled_high_quality.csv
#   - sampled_standard_quality.csv
#   - sampled_low_quality.csv
#   - sampled_best_per_root_language.csv
#   - sampled_summary.json
#
# Note:
#   Quality buckets are DISJOINT:
#   gold ∩ high_quality ∩ standard_quality ∩ low_quality = empty
# ============================================================

import os
import json
import argparse
import numpy as np
import pandas as pd

# =========================
# CLI ARGUMENTS
# =========================

parser = argparse.ArgumentParser(
    description="Filter translation-quality data using BERTScore + COMET thresholds."
)

parser.add_argument(
    "--bert_file",
    required=True,
    help="Path to BERTScore evaluation_table.csv"
)

parser.add_argument(
    "--comet_file",
    required=True,
    help="Path to COMET evaluation_table_comet_full.csv"
)

parser.add_argument(
    "--output_dir",
    required=True,
    help="Directory where output CSV/JSON files will be saved"
)

args = parser.parse_args()

os.makedirs(args.output_dir, exist_ok=True)

# =========================
# FILE PATHS
# =========================

BERT_FILE = args.bert_file
COMET_FILE = args.comet_file

MERGED_OUT = os.path.join(args.output_dir, "merged_scores.csv")
GOLD_OUT = os.path.join(args.output_dir, "sampled_gold.csv")
HIGH_QUALITY_OUT = os.path.join(args.output_dir, "sampled_high_quality.csv")
STANDARD_QUALITY_OUT = os.path.join(args.output_dir, "sampled_standard_quality.csv")
LOW_QUALITY_OUT = os.path.join(args.output_dir, "sampled_low_quality.csv")
BEST_OUT = os.path.join(args.output_dir, "sampled_best_per_root_language.csv")
SUMMARY_OUT = os.path.join(args.output_dir, "sampled_summary.json")

# =========================
# THRESHOLDS
# =========================

GOLD_F1 = 0.95
GOLD_COMET = 0.92
GOLD_COMBINED = 0.94

HIGH_QUALITY_F1 = 0.90
HIGH_QUALITY_COMET = 0.85
HIGH_QUALITY_COMBINED = 0.88

STANDARD_QUALITY_F1 = 0.80
STANDARD_QUALITY_COMET = 0.70
STANDARD_QUALITY_COMBINED = 0.75

# =========================
# LOAD DATA
# =========================

print("Loading data...")

bert_df = pd.read_csv(BERT_FILE)
comet_df = pd.read_csv(COMET_FILE)

comet_df["comet"] = pd.to_numeric(
    comet_df["comet"],
    errors="coerce"
)

# =========================
# MERGE
# =========================

merge_keys = ["root_id", "language"]

merged = bert_df.merge(
    comet_df[merge_keys + ["comet"]],
    on=merge_keys,
    how="left"
)

print(f"Merged rows: {len(merged):,}")

# =========================
# COMBINED SCORE
# =========================

merged["combined_score"] = np.where(
    merged["comet"].notna(),
    (merged["f1"] + merged["comet"]) / 2.0,
    merged["f1"]
)

merged["comet_available"] = merged["comet"].notna().astype(int)

# =========================
# QUALITY BUCKET HELPERS
# =========================

def passes_gold(row):
    if pd.notna(row["comet"]):
        return (
            row["f1"] >= GOLD_F1 and
            row["comet"] >= GOLD_COMET and
            row["combined_score"] >= GOLD_COMBINED
        )
    else:
        return row["f1"] >= GOLD_F1


def passes_high_quality(row):
    if pd.notna(row["comet"]):
        return (
            row["f1"] >= HIGH_QUALITY_F1 and
            row["comet"] >= HIGH_QUALITY_COMET and
            row["combined_score"] >= HIGH_QUALITY_COMBINED
        )
    else:
        return row["f1"] >= HIGH_QUALITY_F1


def passes_standard_quality(row):
    if pd.notna(row["comet"]):
        return (
            row["f1"] >= STANDARD_QUALITY_F1 and
            row["comet"] >= STANDARD_QUALITY_COMET and
            row["combined_score"] >= STANDARD_QUALITY_COMBINED
        )
    else:
        return row["f1"] >= STANDARD_QUALITY_F1


def quality_bucket(row):
    if passes_gold(row):
        return "gold"

    elif passes_high_quality(row):
        return "high_quality"

    elif passes_standard_quality(row):
        return "standard_quality"

    else:
        return "low_quality"


merged["quality_bucket"] = merged.apply(
    quality_bucket,
    axis=1
)

# =========================
# FILTERING
# =========================

print("Applying disjoint filters...")

filtered_gold = merged.loc[
    merged["quality_bucket"] == "gold"
].copy()

filtered_high_quality = merged.loc[
    merged["quality_bucket"] == "high_quality"
].copy()

filtered_standard_quality = merged.loc[
    merged["quality_bucket"] == "standard_quality"
].copy()

filtered_low_quality = merged.loc[
    merged["quality_bucket"] == "low_quality"
].copy()

# =========================
# BEST PER ROOT + LANGUAGE
# =========================

best_per_root_language = (
    merged.sort_values(
        ["root_id", "language", "combined_score"],
        ascending=[True, True, False]
    )
    .drop_duplicates(
        subset=["root_id", "language"],
        keep="first"
    )
    .copy()
)

# =========================
# SAVE FILES
# =========================

print("Saving outputs...")

merged.to_csv(MERGED_OUT, index=False)
filtered_gold.to_csv(GOLD_OUT, index=False)
filtered_high_quality.to_csv(HIGH_QUALITY_OUT, index=False)
filtered_standard_quality.to_csv(STANDARD_QUALITY_OUT, index=False)
filtered_low_quality.to_csv(LOW_QUALITY_OUT, index=False)
best_per_root_language.to_csv(BEST_OUT, index=False)

# =========================
# SUMMARY
# =========================

summary = {
    "total_rows": int(len(merged)),

    "gold_kept": int(len(filtered_gold)),
    "high_quality_kept": int(len(filtered_high_quality)),
    "standard_quality_kept": int(len(filtered_standard_quality)),
    "low_quality_kept": int(len(filtered_low_quality)),

    "gold_ratio": float(len(filtered_gold) / len(merged)),
    "high_quality_ratio": float(len(filtered_high_quality) / len(merged)),
    "standard_quality_ratio": float(len(filtered_standard_quality) / len(merged)),
    "low_quality_ratio": float(len(filtered_low_quality) / len(merged)),

    "quality_bucket_counts": (
        merged["quality_bucket"]
        .value_counts()
        .to_dict()
    ),

    "bucket_definition": "disjoint",

    "thresholds": {
        "gold": {
            "f1": GOLD_F1,
            "comet": GOLD_COMET,
            "combined_score": GOLD_COMBINED
        },
        "high_quality": {
            "f1": HIGH_QUALITY_F1,
            "comet": HIGH_QUALITY_COMET,
            "combined_score": HIGH_QUALITY_COMBINED
        },
        "standard_quality": {
            "f1": STANDARD_QUALITY_F1,
            "comet": STANDARD_QUALITY_COMET,
            "combined_score": STANDARD_QUALITY_COMBINED
        }
    }
}

with open(SUMMARY_OUT, "w") as f:
    json.dump(summary, f, indent=2)

# =========================
# LOG
# =========================

print("\n===== FINAL SUMMARY =====")
print(f"Total rows: {len(merged):,}")

print(
    f"Gold: {len(filtered_gold):,} "
    f"({len(filtered_gold)/len(merged):.2%})"
)

print(
    f"High-Quality: {len(filtered_high_quality):,} "
    f"({len(filtered_high_quality)/len(merged):.2%})"
)

print(
    f"Standard-Quality: {len(filtered_standard_quality):,} "
    f"({len(filtered_standard_quality)/len(merged):.2%})"
)

print(
    f"Low-Quality: {len(filtered_low_quality):,} "
    f"({len(filtered_low_quality)/len(merged):.2%})"
)

print(f"Best per Root+Language: {len(best_per_root_language):,}")

print("\nBucket check:")
print(
    "Total bucket rows:",
    len(filtered_gold)
    + len(filtered_high_quality)
    + len(filtered_standard_quality)
    + len(filtered_low_quality)
)

print("\nOutput files:")
print(f"Merged: {MERGED_OUT}")
print(f"Gold: {GOLD_OUT}")
print(f"High-Quality: {HIGH_QUALITY_OUT}")
print(f"Standard-Quality: {STANDARD_QUALITY_OUT}")
print(f"Low-Quality: {LOW_QUALITY_OUT}")
print(f"Best per Root+Language: {BEST_OUT}")
print(f"Summary: {SUMMARY_OUT}")

print("\nDONE")