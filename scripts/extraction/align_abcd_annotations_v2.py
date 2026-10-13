#!/usr/bin/env python3
"""Align ABCD Atlas.ti annotations to layout-aware PDF pages.

Run from abcd-pubs-mining with the SAME `python` that has pymupdf4llm:

    python scripts/extraction/align_abcd_annotations_v2.py

Only three diagnostic papers are processed by default. Use --all to process
all publications, after checking the pilot results. Input CSV and existing
checkpoints are NEVER modified.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import html
import json
import re
import sys
import time
import unicodedata
from collections import Counter, defaultdict
from difflib import SequenceMatcher
from pathlib import Path

CSV_DEFAULT = Path("data/raw/annotations-cleaned-2026-10-9.csv")
PDF_DEFAULT = Path("/Users/elliephan/Downloads/abcd-proj")
OUTPUT_DEFAULT = Path("alignment_v2_pilot.jsonl")
CACHE_DEFAULT = Path("data/cache/pdf_layout_v2")
PILOT_PREFIXES = ("Tomasi-2021-", "Chiesa-2024-", "Dall'Aglio-2022-")
CONTEXT_WINDOW = 1000
PARSER_VERSION = "pymupdf4llm-pagechunks-v1"
TOKEN_PATTERN = re.compile(r"[^\W_]+", flags=re.UNICODE)


def clean_text(text: str) -> str:
    """Strip PDF Markdown/HTML artifacts, retaining numeric research values."""
    text = unicodedata.normalize("NFKC", html.unescape(str(text or "")))
    text = re.sub(r"<!--.*?-->", " ", text, flags=re.DOTALL)
    # Preserve words separated by a discretionary line-break hyphen.
    text = re.sub(r"(?<=\w)-\s*(?:<br\s*/?>|\n)\s*(?=\w)", "", text, flags=re.I)
    text = re.sub(r"<br\s*/?>", "\n", text, flags=re.I)
    text = re.sub(r"<[^>]*>", " ", text)
    text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)
    text = re.sub(r"(?m)^\s{0,3}#{1,6}\s*", "", text)
    text = re.sub(r"[*_`]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def tokens_with_spans(text: str):
    """Tokens plus offsets back to CLEANED real PDF text."""
    tokens, spans = [], []
    for match in TOKEN_PATTERN.finditer(text):
        t = match.group().casefold()
        tokens.append(t)
        spans.append(match.span())
    return tokens, spans


def build_page_record(raw_text: str, page_number: int) -> dict:
    text = clean_text(raw_text)
    tokens, spans = tokens_with_spans(text)
    word_index = defaultdict(list)
    grams = defaultdict(list)
    for i, word in enumerate(tokens):
        word_index[word].append(i)
        if i + 4 <= len(tokens):
            grams[tuple(tokens[i:i + 4])].append(i)
    return {"page": page_number, "text": text, "tokens": tokens,
            "spans": spans, "word_index": word_index, "grams": grams}


def all_exact_matches(pages: list[dict], target: list[str]) -> list[tuple]:
    """Exact token matches (punctuation and Markdown differences ignored)."""
    results = []
    n = len(target)
    for p in pages:
        source = p["tokens"]
        if n >= 4:
            starts = p["grams"].get(tuple(target[:4]), [])
        else:
            starts = p["word_index"].get(target[0], [])
        for start in starts:
            if source[start:start + n] == target:
                results.append((p, start, start + n, 1.0))
    return results


def fuzzy_matches(pages: list[dict], target: list[str]) -> list[tuple]:
    """Find plausible locations via intact four-word anchors, then score locally.

    This does NOT perform unrestricted whole-paper fuzzy searching. Candidates
    are marked for review, never silently promoted to gold training matches.
    """
    n = len(target)
    if n < 8:
        return []  # A short label like 'race' cannot be located reliably.
    offsets = sorted({0, n // 4, n // 2, 3 * n // 4, max(0, n - 4)})
    lengths = sorted({max(3, n - 6), max(3, n - 2), n, n + 2, n + 6})
    candidates = []
    for p in pages:
        source = p["tokens"]
        possible_starts = set()
        for offset in offsets:
            if offset + 4 > n:
                continue
            anchor = tuple(target[offset:offset + 4])
            locations = p["grams"].get(anchor, [])
            # Skip uninformative anchors repeated throughout a long paper.
            if len(locations) > 100:
                continue
            for pos in locations:
                for shift in (-2, -1, 0, 1, 2):
                    start = pos - offset + shift
                    if 0 <= start < len(source):
                        possible_starts.add(start)
        for start in sorted(possible_starts):
            best_score, best_end = 0.0, start
            for length in lengths:
                end = min(len(source), start + length)
                if end - start < 4:
                    continue
                # Penalize substitutions, deletions, and extra manuscript line numbers.
                score = SequenceMatcher(None, target, source[start:end], autojunk=False).ratio()
                if score > best_score:
                    best_score, best_end = score, end
            if best_score >= 0.80:
                candidates.append((p, start, best_end, round(best_score, 4)))
    # Remove near-identical shifted candidates; retain one per passage.
    candidates.sort(key=lambda item: item[3], reverse=True)
    distinct = []
    for c in candidates:
        if not any(c[0]["page"] == d[0]["page"] and abs(c[1] - d[1]) < 8 for d in distinct):
            distinct.append(c)
        if len(distinct) >= 10:
            break
    return distinct


def location_details(match: tuple) -> dict:
    p, start, end, score = match
    spans, text = p["spans"], p["text"]
    char_start = spans[start][0]
    char_end = spans[end - 1][1]
    return {
        "page": p["page"],
        "char_start": char_start,
        "char_end": char_end,
        "score": score,
        "matched_pdf_text": text[char_start:char_end],
        "context": text[max(0, char_start - CONTEXT_WINDOW):
                        min(len(text), char_end + CONTEXT_WINDOW)],
    }


def align_annotation(pages: list[dict], excerpt: str) -> dict:
    cleaned = clean_text(excerpt)
    target, _ = tokens_with_spans(cleaned)
    if not target:
        return {"match_status": "empty_annotation", "needs_review": True,
                "usable_for_training": False, "candidate_count": 0}

    matches = all_exact_matches(pages, target)
    if len(matches) == 1:
        return {"match_status": "exact_unique", "needs_review": False,
                "usable_for_training": True, "candidate_count": 1,
                **location_details(matches[0])}
    if len(matches) > 1:
        return {"match_status": "exact_multiple", "needs_review": True,
                "usable_for_training": False, "candidate_count": len(matches),
                "candidate_pages": sorted({m[0]["page"] for m in matches})}

    candidates = fuzzy_matches(pages, target)
    if not candidates:
        return {"match_status": "not_found", "needs_review": True,
                "usable_for_training": False, "candidate_count": 0}

    best = candidates[0]
    # Conservative: only expose high-confidence uniquely ranked suggestions.
    second_score = candidates[1][3] if len(candidates) > 1 else 0.0
    if best[3] >= 0.88 and (best[3] - second_score >= 0.04):
        return {"match_status": "fuzzy_candidate", "needs_review": True,
                "usable_for_training": False, "candidate_count": len(candidates),
                **location_details(best)}
    return {"match_status": "fuzzy_ambiguous", "needs_review": True,
            "usable_for_training": False, "candidate_count": len(candidates),
            "best_candidate_score": best[3],
            "candidate_pages": sorted({m[0]["page"] for m in candidates})}


def read_annotations(path: Path) -> dict[str, list[dict]]:
    papers = defaultdict(list)
    with path.open(newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        required = {"Source", "Content", "Codes", "Annotator"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Missing CSV columns: {sorted(missing)}")
        for index, row in enumerate(reader, start=2):
            source = (row.get("Source") or "").strip()
            excerpt = (row.get("Content") or "").strip()
            if source and excerpt:
                papers[source].append({
                    "csv_row": index, "source": source,
                    "annotation": excerpt,
                    "code": (row.get("Codes") or "").strip(),
                    "annotator": (row.get("Annotator") or "").strip(),
                })
    return papers


def extract_pages_cached(pdf_path: Path, cache_dir: Path) -> list[str]:
    """Run the document parser ONCE per PDF, caching raw page Markdown."""
    stamp = pdf_path.stat()
    identity = f"{pdf_path.resolve()}|{stamp.st_size}|{stamp.st_mtime_ns}|{PARSER_VERSION}"
    key = hashlib.sha256(identity.encode()).hexdigest()
    cache_path = cache_dir / f"{key}.json"
    if cache_path.exists():
        return json.loads(cache_path.read_text(encoding="utf-8"))

    import pymupdf4llm  # Import lazily; run with `python`, not Homebrew `python3`.
    parsed = pymupdf4llm.to_markdown(str(pdf_path), page_chunks=True)
    pages = [str(part.get("text") or "") for part in parsed]
    cache_dir.mkdir(parents=True, exist_ok=True)
    temp = cache_path.with_suffix(".tmp")
    temp.write_text(json.dumps(pages, ensure_ascii=False), encoding="utf-8")
    temp.replace(cache_path)
    return pages


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, default=CSV_DEFAULT)
    parser.add_argument("--pdf-dir", type=Path, default=PDF_DEFAULT)
    parser.add_argument("--output", type=Path, default=OUTPUT_DEFAULT)
    parser.add_argument("--cache-dir", type=Path, default=CACHE_DEFAULT)
    parser.add_argument("--all", action="store_true", help="Process all 1,402 papers")
    parser.add_argument("--papers", nargs="+", help="Optional Source filename prefixes to test")
    args = parser.parse_args()

    if not args.csv.is_file():
        parser.error(f"Annotation CSV not found: {args.csv}")
    if not args.pdf_dir.is_dir():
        parser.error(f"PDF directory not found: {args.pdf_dir}")

    # Uses the PDF filename index already verified in your original project.
    try:
        from match_pdfs_to_annotations import build_pdf_index, find_pdf_for_paper
    except ImportError as e:
        parser.error(f"Could not import your existing PDF index functions: {e}")

    papers = read_annotations(args.csv)
    prefixes = args.papers or PILOT_PREFIXES
    if not args.all:
        papers = {name: rows for name, rows in papers.items()
                  if any(name.startswith(prefix) for prefix in prefixes)}
    if not papers:
        parser.error("No publications matched the requested prefixes")

    index = build_pdf_index(args.pdf_dir)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    totals = Counter()
    started = time.perf_counter()
    with args.output.open("w", encoding="utf-8") as out:
        for number, (source, rows) in enumerate(papers.items(), start=1):
            print(f"[{number}/{len(papers)}] {source}", flush=True)
            path = find_pdf_for_paper(source, index)
            if path is None:
                for row in rows:
                    record = {**row, "match_status": "pdf_not_found",
                              "needs_review": True, "usable_for_training": False,
                              "candidate_count": 0}
                    out.write(json.dumps(record, ensure_ascii=False) + "\n")
                    totals[record["match_status"]] += 1
                continue
            try:
                texts = extract_pages_cached(path, args.cache_dir)
                page_records = [build_page_record(text, i)
                                for i, text in enumerate(texts, 1)]
            except Exception as exc:
                print(f"  Extraction error: {exc}", file=sys.stderr)
                for row in rows:
                    record = {**row, "match_status": "pdf_error",
                              "error": str(exc), "needs_review": True,
                              "usable_for_training": False, "candidate_count": 0}
                    out.write(json.dumps(record, ensure_ascii=False) + "\n")
                    totals[record["match_status"]] += 1
                continue
            counts = Counter()
            for row in rows:
                result = align_annotation(page_records, row["annotation"])
                record = {**row, "pdf_path": str(path), **result}
                out.write(json.dumps(record, ensure_ascii=False) + "\n")
                totals[result["match_status"]] += 1
                counts[result["match_status"]] += 1
            print("  ", dict(counts), flush=True)

    print("\n=== SUMMARY ===")
    print("Publications:", len(papers))
    print("Annotations:", sum(totals.values()))
    print("By status:", dict(totals))
    print("Elapsed seconds:", round(time.perf_counter() - started, 1))
    print("Saved:", args.output)
    print("Only exact_unique records are automatically marked usable_for_training=True.")


if __name__ == "__main__":
    main()
