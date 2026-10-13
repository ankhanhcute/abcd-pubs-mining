"""
match_pdfs_to_annotations.py
------------------------------
Matches each paper's annotated excerpts (Findings, IV, DV, etc.) to the actual
PDF text and extracts the REAL surrounding paragraph as training context.

Handles mixed one-column / two-column PDFs by trying several extractors per
paper and keeping the first one that finds the excerpt exactly once:

    1. layout       (pymupdf4llm, layout-aware, handles most column layouts)
    2. two_columns  (pdfplumber, splits each page down the middle)
    3. plain        (pdfplumber, no column handling)

Each output row records which extractor matched, so you can see empirically
which one works for your papers.

match_status values:
    matched            excerpt found exactly once by some extractor
    ambiguous_match    excerpt found, but more than once (no safe location)
    excerpt_not_found  no extractor found the excerpt
    pdf_not_found      no PDF for this paper

USAGE
-----
    python3 match_pdfs_to_annotations.py \
        --csv marian_export.csv \
        --pdf-dir ./abcd-pubs-ai \
        --output training_pairs_with_context.jsonl

REQUIRES: pip install pdfplumber pymupdf4llm
"""

import argparse
import csv
import hashlib
import json
import os
import re
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

import pdfplumber
import pymupdf4llm


CONTEXT_WINDOW = 1000  # characters before/after the matched excerpt


# --------------------------------------------------------------------------
# Text normalization
# --------------------------------------------------------------------------

def clean_text(text: str) -> str:
    """Normalize text so PDF text and annotator excerpts can be compared.

    Keeps original casing (lowercase for searching happens separately).
    """
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("\u2018", "'").replace("\u2019", "'")
    text = text.replace("\u201c", '"').replace("\u201d", '"')
    text = text.replace("\u2013", "-").replace("\u2014", "-").replace("\u2212", "-")
    text = text.replace("\u00ad", "")                  # soft hyphen
    text = re.sub(r"-\s*\n\s*", "", text)               # join words split across lines
    text = re.sub(r"[*_`#]", "", text)                  # markdown emphasis/headers (pymupdf4llm)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def prepare_text(text: str):
    """Return (display_text, search_text). Same length so indices line up."""
    display = clean_text(text)
    search = display.lower()
    if len(search) != len(display):
        # Rare: lowercasing changed length. Fall back to lowercase for both.
        display = search
    return display, search


# --------------------------------------------------------------------------
# Matching
# --------------------------------------------------------------------------

def find_context_window(display: str, search: str, excerpt: str,
                        window: int = CONTEXT_WINDOW):
    """Look for the excerpt in the prepared PDF text.

    Returns (status, context) where status is
    "matched", "ambiguous", or "not_found".
    """
    target = clean_text(excerpt).lower()
    if not target:
        return "not_found", None

    position = search.find(target)
    if position == -1:
        return "not_found", None

    if search.find(target, position + 1) != -1:
        return "ambiguous", None

    start = max(0, position - window)
    end = min(len(display), position + len(target) + window)
    return "matched", display[start:end]


# --------------------------------------------------------------------------
# Loading annotations and finding PDFs
# --------------------------------------------------------------------------

def load_annotations(csv_path: str) -> dict:
    """Group CSV rows by paper (Source)."""
    papers = defaultdict(list)

    with open(csv_path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        required = {"Source", "Content", "Codes", "Annotator"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"CSV is missing columns: {sorted(missing)}")

        for row in reader:
            source = (row["Source"] or "").strip()
            content = (row["Content"] or "").strip()
            code = (row["Codes"] or "").strip()
            annotator = (row["Annotator"] or "").strip()

            if not source or not content or not code:
                continue
            papers[source].append({
                "content": content,
                "code": code,
                "annotator": annotator,
            })
    return dict(papers)


def normalize_filename(name):
    name = Path(name).name
    if name.lower().endswith(".pdf"):
        name = name[:-4]
    name = re.sub(r"\s*\(\d+\)$", "", name)  # drop duplicate suffixes like (1)
    return re.sub(r"[^a-z0-9]", "", name.lower())


def build_pdf_index(pdf_dir):
    """Scan the PDF collection once, following Google Drive links."""
    pdf_dir = Path(pdf_dir).expanduser()
    if not pdf_dir.is_dir():
        raise ValueError(f"PDF folder does not exist: {pdf_dir}")

    index = defaultdict(list)
    for root, _dirs, files in os.walk(pdf_dir, followlinks=True):
        for filename in files:
            if filename.lower().endswith(".pdf"):
                index[normalize_filename(filename)].append(Path(root) / filename)

    print(f"Indexed {sum(len(v) for v in index.values())} PDFs")
    return index


def file_hash(path):
    sha = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            sha.update(chunk)
    return sha.hexdigest()


def find_pdf_for_paper(paper_id, pdf_index):
    matches = pdf_index.get(normalize_filename(paper_id), [])
    if not matches:
        return None

    expected_name = Path(paper_id).name
    if not expected_name.lower().endswith(".pdf"):
        expected_name += ".pdf"

    exact = [p for p in matches if p.name.lower() == expected_name.lower()]
    if len(exact) == 1:
        return exact[0]

    candidates = exact if exact else matches
    if len(candidates) == 1:
        return candidates[0]

    # Several copies: fine if they are byte-identical, otherwise ambiguous.
    try:
        hashes = {file_hash(p) for p in candidates}
    except OSError as e:
        print(f"  WARNING: Cannot read duplicate PDF: {e}")
        return None

    if len(hashes) == 1:
        return sorted(candidates)[0]

    print(f"  WARNING: Ambiguous PDF copies for {paper_id}")
    return None


# --------------------------------------------------------------------------
# PDF text extractors
# --------------------------------------------------------------------------

def extract_pdf_text(pdf_path: Path) -> str:
    """Plain extraction, no column handling."""
    text_parts = []
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text(x_tolerance=1)
            if page_text:
                text_parts.append(page_text)
    return "\n".join(text_parts)


def extract_pdf_text_two_columns(pdf_path: Path) -> str:
    """Split each page down the middle: left column, then right column.

    Assumes every page is two-column. Full-width titles, figures and tables
    can get cut, which is why this is not the only extractor tried.
    """
    text_parts = []
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            midpoint = page.width / 2
            left = page.crop((0, 0, midpoint, page.height))
            right = page.crop((midpoint, 0, page.width, page.height))

            left_text = left.extract_text(x_tolerance=1) or ""
            right_text = right.extract_text(x_tolerance=1) or ""

            page_text = "\n".join(p for p in [left_text, right_text] if p.strip())
            text_parts.append(page_text)
    return "\n".join(text_parts)


def extract_pdf_text_layout(pdf_path: Path) -> str:
    """Layout-aware extraction (outputs markdown; clean_text strips the markup)."""
    pages = pymupdf4llm.to_markdown(str(pdf_path), page_chunks=True)
    return "\n".join(page["text"] for page in pages)


# Tried in this order; first extractor to find a unique match wins.
EXTRACTORS = [
    ("layout", extract_pdf_text_layout),
    ("two_columns", extract_pdf_text_two_columns),
    ("plain", extract_pdf_text),
]


class PaperText:
    """Lazily extracts and caches a paper's text for each extractor."""

    def __init__(self, pdf_path: Path):
        self.pdf_path = pdf_path
        self._cache = {}

    def get(self, name, fn):
        if name not in self._cache:
            try:
                self._cache[name] = prepare_text(fn(self.pdf_path))
            except Exception as e:
                print(f"  WARNING: {name} extractor failed on "
                      f"{self.pdf_path.name}: {e}")
                self._cache[name] = ("", "")
        return self._cache[name]


def match_annotation(paper_text: PaperText, excerpt: str):
    """Try each extractor in order.

    Returns (status, context, extractor_name).
    """
    saw_ambiguous = False

    for name, fn in EXTRACTORS:
        display, search = paper_text.get(name, fn)
        if not search:
            continue

        status, context = find_context_window(display, search, excerpt)

        if status == "matched":
            return "matched", context, name
        if status == "ambiguous":
            saw_ambiguous = True

    if saw_ambiguous:
        return "ambiguous_match", None, None
    return "excerpt_not_found", None, None


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Match ABCD annotation categories to PDF context"
    )
    parser.add_argument("--csv", required=True)
    parser.add_argument("--pdf-dir", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    papers = load_annotations(args.csv)
    pdf_index = build_pdf_index(args.pdf_dir)

    # One checkpoint file per publication, so a crash only loses one paper.
    checkpoint_dir = Path(args.output + ".checkpoints")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    status_counts = Counter()
    extractor_counts = Counter()
    pdf_not_found_papers = 0

    for i, (paper_id, annotations) in enumerate(papers.items(), 1):
        checkpoint_id = hashlib.sha256(paper_id.encode("utf-8")).hexdigest()
        checkpoint = checkpoint_dir / f"{checkpoint_id}.jsonl"

        if checkpoint.exists():
            print(f"[{i}/{len(papers)}] Already processed: {paper_id}")
            continue

        print(f"[{i}/{len(papers)}] Processing: {paper_id}")

        pdf_path = find_pdf_for_paper(paper_id, pdf_index)
        paper_text = PaperText(pdf_path) if pdf_path else None
        if pdf_path is None:
            pdf_not_found_papers += 1

        paper_pairs = []

        for annotation in annotations:
            excerpt = annotation["content"]

            if paper_text is None:
                status, context, extractor = "pdf_not_found", None, None
            else:
                status, context, extractor = match_annotation(paper_text, excerpt)

            if context is None:
                context = excerpt  # fallback: annotator's excerpt alone

            status_counts[status] += 1
            if extractor:
                extractor_counts[extractor] += 1

            paper_pairs.append({
                "source": paper_id,
                "annotator": annotation["annotator"],
                "code": annotation["code"],
                "annotation": excerpt,
                "context": context,
                "match_status": status,
                "extractor": extractor,
            })

        # Write to a temp file, then rename, so a checkpoint is never half-written.
        temporary = checkpoint.with_suffix(".tmp")
        with open(temporary, "w", encoding="utf-8") as f:
            for pair in paper_pairs:
                f.write(json.dumps(pair, ensure_ascii=False) + "\n")
        temporary.replace(checkpoint)

    # Combine all completed publications into the final JSONL.
    output = Path(args.output)
    with open(output, "w", encoding="utf-8") as outfile:
        for paper_id in papers:
            checkpoint_id = hashlib.sha256(paper_id.encode("utf-8")).hexdigest()
            checkpoint = checkpoint_dir / f"{checkpoint_id}.jsonl"
            if not checkpoint.exists():
                continue
            with open(checkpoint, encoding="utf-8") as infile:
                for line in infile:
                    outfile.write(line)

    print("\n=== SUMMARY (this run only) ===")
    print(f"Publications in dataset: {len(papers)}")
    print(f"Publications without PDF (new): {pdf_not_found_papers}")
    for status in ["matched", "ambiguous_match", "excerpt_not_found", "pdf_not_found"]:
        print(f"{status}: {status_counts[status]}")
    print("Matches by extractor:")
    for name, _ in EXTRACTORS:
        print(f"  {name}: {extractor_counts[name]}")
    print(f"Output: {output}")
    print(f"Checkpoints: {checkpoint_dir}")


if __name__ == "__main__":
    main()