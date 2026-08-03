"""
match_pdfs_to_annotations.py
------------------------------
Matches each paper's annotated Findings/IV/DV excerpts to the actual PDF
text, and extracts the REAL surrounding paragraph as training context --
instead of using just the clean one-line excerpt from Atlas.ti.
 
WHY THIS MATTERS
----------------
build_training_pairs.py (the earlier version) used the annotator's short
Findings sentence as the prompt. That's a clean proxy, but it's not what
the model will actually see in production -- real papers, with messy
surrounding text. This script fixes that by finding the real paragraph
in the actual PDF around where each annotation was pulled from.
 
HOW MATCHING WORKS
-------------------
1. For each paper (Source column in the CSV), find the matching PDF file
   by filename (the Source value should match the PDF filename, e.g.
   "Adams-2026-....pdf").
2. Extract all text from that PDF.
3. Search for the annotator's Findings excerpt (or a close match, since
   PDF text extraction sometimes has minor whitespace/formatting
   differences) within the full text.
4. If found, pull a window of text around it (e.g. 500 characters before
   and after) as the real paragraph context.
5. If NOT found (extraction differences, OCR issues, etc.), fall back to
   using just the annotator's excerpt alone -- logged so you know which
   papers needed the fallback.
 
USAGE
-----
    python3 match_pdfs_to_annotations.py \\
        --csv marian_export.csv \\
        --pdf-dir ./abcd-pubs-ai \\
        --output training_pairs_with_context.jsonl
 
REQUIRES: pip install pdfplumber
"""


import argparse
import csv 
import json
import re
from collections import defaultdict
from pathlib import Path 

import pdfplumber


CONTEXT_WINDOW = 750 #characters bf/after the matched excerpt

def load_annotations(csv_path: str) -> dict:
    papers = defaultdict(lambda: {"iv": [], "dv": [], "findings": []})
    with open(csv_path, "r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            src = row["Source"].strip()
            code = row["Codes"].strip()
            content = row["Text Content"].strip()
            if not src or not content:
                continue
            if code == "Independent Variable":
                papers[src]["iv"].append(content)
            elif code == "Dependent Variable":
                papers[src]["dv"].append(content)
            elif code == "Findings":
                papers[src]["findings"].append(content)
                
    return papers

def find_pdf_for_papers(paper_id: str, pdf_dir: Path) -> Path | None:
    """
    Find the PDF File matching by the paper id
    
    Tries exact match first and then loose dmatch in case of minor files differentesd extra
    spaces,  missing .pdf extension in the csv, etc.)
    """
    
    exact = pdf_dir / f"{paper_id}.pdf" if not paper_id.endswith(".pdf") else pdf_dir / paper_dir
    
    if exact.exists():
        return exact
    
    #normalize bothside lowercase strip non-alphanumerics:
    def normalize(s):
        return re.sub(r"[^a-z0-9]", "", s.lower())
    
    target = normalize(paper_id)
    for pdf_file in pdf_dir.glob("*.pdf"):
        if normalize(pdf_file.stem) == target:
            return pdf_file
    return None
def extract_pdf_text(pdf_path: Path) -> str:
    try:
        text_parts = []
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text_parts.append(page_text)
        return "\n".join(text_parts)
    except Exception as e:
        print(f"  WARNING: failed to extract text from {pdf_path.name}: {e}")
        return ""
 
 
def normalize_for_search(text: str) -> str:
    """Collapse whitespace so PDF extraction quirks don't break matching."""
    return re.sub(r"\s+", " ", text).strip()
 
 
def find_context_window(full_text: str, excerpt: str, window: int = CONTEXT_WINDOW) -> str | None:
    """Find the excerpt in the full PDF text and return surrounding context.
 
    Returns None if the excerpt can't be found (fallback needed).
    """
    normalized_full = normalize_for_search(full_text)
    normalized_excerpt = normalize_for_search(excerpt)
 
    # Try exact match first
    idx = normalized_full.find(normalized_excerpt)
 
    # If exact match fails, try matching on just the first ~10 words --
    # PDF extraction sometimes garbles longer excerpts partway through.
    if idx == -1:
        words = normalized_excerpt.split()
        if len(words) > 10:
            partial = " ".join(words[:10])
            idx = normalized_full.find(partial)
 
    if idx == -1:
        return None
 
    start = max(0, idx - window)
    end = min(len(normalized_full), idx + len(normalized_excerpt) + window)
    return normalized_full[start:end]
 
 
def build_prompt(context_text: str) -> str:
    return (
        "Extract the independent variable(s) (IV) and dependent variable(s) (DV) "
        f"from this excerpt of a research paper.\n\nExcerpt: \"{context_text}\""
    )
 
 
def main():
    parser = argparse.ArgumentParser(description="Match annotations to real PDF context")
    parser.add_argument("--csv", required=True, help="Path to raw Atlas.ti CSV export")
    parser.add_argument("--pdf-dir", required=True, help="Path to folder containing paper PDFs")
    parser.add_argument("--output", required=True, help="Path to write output JSONL")
    args = parser.parse_args()
 
    pdf_dir = Path(args.pdf_dir)
    papers = load_annotations(args.csv)
 
    pairs = []
    matched_count = 0
    fallback_count = 0
    pdf_not_found_count = 0
 
    for i, (paper_id, data) in enumerate(papers.items(), 1):
        ivs, dvs, findings = data["iv"], data["dv"], data["findings"]
        if not (ivs and dvs and findings):
            continue
 
        excerpt = findings[0]
        print(f"[{i}/{len(papers)}] {paper_id}")
 
        pdf_path = find_pdf_for_paper(paper_id, pdf_dir)
        if pdf_path is None:
            print(f"  PDF not found for this paper -- using excerpt-only fallback")
            pdf_not_found_count += 1
            context = excerpt
        else:
            full_text = extract_pdf_text(pdf_path)
            context = find_context_window(full_text, excerpt) if full_text else None
            if context:
                matched_count += 1
            else:
                print(f"  Excerpt not found in PDF text -- using excerpt-only fallback")
                fallback_count += 1
                context = excerpt
 
        for iv in ivs:
            for dv in dvs:
                pairs.append({
                    "prompt": build_prompt(context),
                    "completion": f"IV: {iv}\nDV: {dv}",
                })
 
    with open(args.output, "w", encoding="utf-8") as f:
        for pair in pairs:
            f.write(json.dumps(pair) + "\n")
 
    print(f"\n=== Summary ===")
    print(f"Papers processed: {len(papers)}")
    print(f"Matched to real PDF context: {matched_count}")
    print(f"Fallback to excerpt-only (PDF found but text not matched): {fallback_count}")
    print(f"PDF file not found at all: {pdf_not_found_count}")
    print(f"Total training pairs written: {len(pairs)}")
    print(f"Output: {args.output}")
 
 
if __name__ == "__main__":
    main()
 
