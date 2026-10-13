
import sys
import csv
import random
from pathlib import Path
from collections import Counter

# Import our existing PDF extraction functions
sys.path.insert(0, str(Path(__file__).parent))

from match_pdfs_to_annotations import (
    build_pdf_index,
    find_pdf_for_paper,
    extract_pdf_text,
    extract_pdf_text_two_columns,
    normalize_for_search,
)

CSV_PATH = Path("data/raw/annotations-cleaned-2026-10-9.csv")
PDF_DIR = Path("/Users/elliephan/Downloads/abcd-proj")

SAMPLE_SIZE = 10
RANDOM_SEED = 42


def evaluate_matching(text, annotations):
    """Count unique, multiple, and missing matches."""
    normalized_text = normalize_for_search(text)
    results = Counter()

    for row in annotations:
        excerpt = normalize_for_search(row["Content"])

        if not excerpt:
            results["empty"] += 1
            continue

        occurrences = normalized_text.count(excerpt)

        if occurrences == 0:
            results["not_found"] += 1
        elif occurrences == 1:
            results["unique"] += 1
        else:
            results["multiple"] += 1

    return results


def main():
    random.seed(RANDOM_SEED)

    # Load human annotations grouped by publication
    papers = {}

    with open(CSV_PATH, encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)

        for row in reader:
            source = row["Source"].strip()

            if source:
                papers.setdefault(source, []).append(row)

    # Index all PDFs just once
    index = build_pdf_index(PDF_DIR)

    # Select only publications with available PDFs
    available = []

    for paper in sorted(papers):
        path = find_pdf_for_paper(paper, index)

        if path is not None:
            available.append((paper, path))

    selected = random.sample(
        available,
        min(SAMPLE_SIZE, len(available))
    )

    original_total = Counter()
    columns_total = Counter()

    print("\n=== PDF EXTRACTION COMPARISON ===")

    for i, (paper, path) in enumerate(selected, 1):
        print(f"\n[{i}/{len(selected)}] {paper}")

        annotations = papers[paper]

        original = extract_pdf_text(path)
        columns = extract_pdf_text_two_columns(path)

        original_results = evaluate_matching(original, annotations)
        columns_results = evaluate_matching(columns, annotations)

        original_total.update(original_results)
        columns_total.update(columns_results)

        print("  Human annotations:", len(annotations))
        print("  Original:", dict(original_results))
        print("  Two columns:", dict(columns_results))

    print("\n=== OVERALL RESULTS ===")
    print("Original:", dict(original_total))
    print("Two columns:", dict(columns_total))

    original_found = (
        original_total["unique"] + original_total["multiple"]
    )
    columns_found = (
        columns_total["unique"] + columns_total["multiple"]
    )

    total = sum(original_total.values())

    print("\n=== MATCHING RATES ===")

    if total:
        print(f"Original: {original_found}/{total} ({original_found/total:.1%})")
        print(f"Two columns: {columns_found}/{total} ({columns_found/total:.1%})")

    print("\nUnique matches:")
    print("Original:", original_total["unique"])
    print("Two columns:", columns_total["unique"])


if __name__ == "__main__":
    main()
