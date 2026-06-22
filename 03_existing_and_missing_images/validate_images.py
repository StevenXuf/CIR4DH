"""
IMAGE EXISTENCE VALIDATION
===========================
Checks which id1 (reference) and id2 (target) image IDs from the CSV
actually exist as .png files in the test_id/ and test_ood/ folders.

Saves two CSV files to:
  03_existing_and_missing_images/results/
    - existing_images.csv   — IDs found on disk
    - missing_images.csv    — IDs not found on disk
"""

from pathlib import Path
import pandas as pd

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG — adjust paths if needed
# ─────────────────────────────────────────────────────────────────────────────
CSV_PATH = Path(
    "/home/users/ymathansingh/Thesis project/02_translation_CIR/"
    "round2/results_round2/v5_do_sample=false/cir_db_translated_qwen_v5.csv"
)

IMAGE_DIRS = [
    Path("/home/users/ymathansingh/Thesis project/test_id"),
    Path("/home/users/ymathansingh/Thesis project/test_ood"),
]

OUT_DIR = Path(
    "/home/users/ymathansingh/Thesis project/"
    "03_existing_and_missing_images/results"
)

# ─────────────────────────────────────────────────────────────────────────────

def find_image(image_id: str) -> Path | None:
    """Return the path if image_id.png exists in any IMAGE_DIR, else None."""
    for folder in IMAGE_DIRS:
        p = folder / f"{image_id}.png"
        if p.exists():
            return p
    return None


def collect_ids(series: pd.Series) -> list[str]:
    """
    Extract unique image IDs from a column.
    id2 can contain comma-separated multiple targets — handles both.
    """
    ids = set()
    for val in series.dropna():
        for iid in str(val).split(","):
            iid = iid.strip()
            if iid:
                ids.add(iid)
    return sorted(ids)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # ── Load CSV ──────────────────────────────────────────────────────────────
    print(f"Loading: {CSV_PATH}")
    df = pd.read_csv(CSV_PATH)
    print(f"  Total rows : {len(df)}")
    print(f"  Partitions : {df['partition'].value_counts().to_dict()}")

    # ── Collect unique IDs ────────────────────────────────────────────────────
    id1_all = collect_ids(df["id1"])
    id2_all = collect_ids(df["id2"])

    print(f"\n  Unique id1 (reference) : {len(id1_all)}")
    print(f"  Unique id2 (target)    : {len(id2_all)}")

    # ── Check existence ───────────────────────────────────────────────────────
    print("\nChecking image files on disk...")

    rows_existing = []
    rows_missing  = []

    for column, ids in [("id1", id1_all), ("id2", id2_all)]:
        found_count   = 0
        missing_count = 0
        for iid in ids:
            p = find_image(iid)
            if p is not None:
                found_count += 1
                rows_existing.append({
                    "column"   : column,
                    "image_id" : iid,
                    "found_in" : str(p.parent.name),   # test_id or test_ood
                    "full_path": str(p),
                })
            else:
                missing_count += 1
                rows_missing.append({
                    "column"   : column,
                    "image_id" : iid,
                })

        print(f"\n  [{column}]  found: {found_count}  |  missing: {missing_count}  "
              f"|  total unique: {len(ids)}")

    # ── Save CSVs ─────────────────────────────────────────────────────────────
    existing_df = pd.DataFrame(rows_existing)
    missing_df  = pd.DataFrame(rows_missing)

    existing_path = OUT_DIR / "existing_images.csv"
    missing_path  = OUT_DIR / "missing_images.csv"

    existing_df.to_csv(existing_path, index=False)
    missing_df.to_csv(missing_path,  index=False)

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    for column in ["id1", "id2"]:
        e = existing_df[existing_df["column"] == column]
        m = missing_df[missing_df["column"] == column]
        total = len(e) + len(m)
        pct_found   = len(e) / total * 100 if total else 0
        pct_missing = len(m) / total * 100 if total else 0

        print(f"\n  {column}:")
        print(f"    Total unique IDs : {total}")
        print(f"    Found on disk    : {len(e):>5}  ({pct_found:.1f}%)")
        print(f"    Missing          : {len(m):>5}  ({pct_missing:.1f}%)")

        if not e.empty:
            by_folder = e.groupby("found_in").size()
            for folder, cnt in by_folder.items():
                print(f"      └─ in {folder}: {cnt}")

    print(f"\n  Saved → {existing_path}")
    print(f"  Saved → {missing_path}")

    # ── Per-partition breakdown for id2 (most relevant for CLIP) ─────────────
    print("\n" + "=" * 60)
    print("ID2 MISSING — BREAKDOWN BY CATEGORY PREFIX")
    print("=" * 60)
    m_id2 = missing_df[missing_df["column"] == "id2"]["image_id"]
    if not m_id2.empty:
        prefixes = m_id2.str.extract(r"^([a-zA-Z_]+)")[0].value_counts()
        for prefix, cnt in prefixes.items():
            print(f"  {prefix:<35} {cnt}")
    else:
        print("  None missing for id2!")

    print("\nDone.")


if __name__ == "__main__":
    main()