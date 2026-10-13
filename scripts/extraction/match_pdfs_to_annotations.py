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

For each excerpt, increasingly forgiving match methods are tried:

    exact     whole-word match after whitespace/quote/dash cleanup
    loose     letters and digits only (ignores spacing, hyphenation,
              punctuation and dropped fi/fl/ff ligatures)
    fragment  excerpt is several pieces joined by newlines; each piece is
              matched on its own
    fuzzy     best approximate match (typos, words changed during cleaning)

Each output row records the extractor and method that matched, so you can see
empirically what works for your papers.

match_status values:
    matched            excerpt found exactly once (exact or loose)
    matched_partial    only a fragment of a multi-piece excerpt was found
    matched_fuzzy      approximate match; "score" (0-100) says how close
    matched_anchored   excerpt found more than once, but another located
                       excerpt from the same paper says which occurrence
    ambiguous_match    excerpt found more than once and nothing pins it down;
                       context is a best guess
    excerpt_not_found  nothing matched; context falls back to the excerpt
    pdf_not_found      no PDF for this paper

For excerpts found more than once, "n_occurrences" says how many times and
"resolved_by" says how the occurrence used for the context was chosen:
    inside_excerpt        it lies inside another located excerpt (reliable)
    near_supporting_info  it is the closest one to the paper's located
                          IV/DV Supporting Info excerpt
    cue_words             its surroundings mention words typical for the code
                          (e.g. "adjusted for" near a Covariate); a guess
    first_occurrence      nothing to go on; first place it appears

USAGE
-----
    python3 match_pdfs_to_annotations.py \
        --csv marian_export.csv \
        --pdf-dir ./abcd-pubs-ai \
        --output training_pairs_with_context.jsonl

Extracted PDF text is cached in <output>.text_cache, so only the first run is
slow. Papers already in <output>.checkpoints are skipped; pass --rerun after
changing this script or the CSV, otherwise the old results are reused.

REQUIRES: pip install pdfplumber pymupdf4llm rapidfuzz
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
from rapidfuzz import fuzz


CONTEXT_WINDOW = 1000  # characters before/after the matched excerpt
MIN_LOOSE_LENGTH = 15  # shortest excerpt (letters/digits) matched ignoring spacing
MIN_FRAGMENT_LENGTH = 20
MIN_FUZZY_LENGTH = 30
FUZZY_THRESHOLD = 90   # 0-100 similarity needed to accept an approximate match
NEAR_ANCHOR_DISTANCE = 2000  # max characters between a label and its Supporting Info
CUE_WINDOW = 200       # characters before/after an occurrence searched for cue words

# A short label and the longer excerpt that describes it.
PAIRED_CODE = {
    "Independent Variable": "IV Supporting Info",
    "Dependent Variable": "DV Supporting Info",
}

# Words that tend to surround a code's label where the paper actually defines it.
CUE_WORDS = {
    "Covariate": ["covariate", "adjust", "controlled for", "controlling for",
                  "control for", "confound"],
    "Independent Variable": ["predictor", "independent variable", "exposure",
                             "was measured", "were measured", "assessed"],
    "Dependent Variable": ["outcome", "dependent variable", "was measured",
                           "were measured", "assessed"],
    "Sample Size": ["sample", "participants", "included", "n ="],
    "ABCD Time Point": ["follow-up", "wave", "time point", "collected", "release"],
    "ABCD Release": ["release", "abcd", "version"],
    "Statistical Model": ["model", "regression", "analys", "estimated"],
    "Participant Age": ["years", "aged", "mean age", "age range"],
    "Participant": ["participants", "recruited", "sample", "excluded"],
}


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
    text = re.sub(r"<br\s*/?>", " ", text)              # html line breaks (pymupdf4llm tables)
    text = re.sub(r"</?(?:sup|sub|mark|u|b|i|s)>", "", text)  # other html tags (pymupdf4llm)
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


# Annotators' copies often lose ligatures ("identifed" for "identified"), so
# every f-ligature collapses to a single "f" on both sides.
LOOSE_TOKEN = re.compile(r"ffi|ffl|ff|fi|fl|[^\W_]")


def loose_key(search: str):
    """Reduce text to letters and digits only.

    Returns (key, positions) where positions[i] is the index in `search`
    of the character that produced key[i].
    """
    chars, positions = [], []
    for token in LOOSE_TOKEN.finditer(search):
        chars.append("f" if len(token.group()) > 1 else token.group())
        positions.append(token.start())
    return "".join(chars), positions


class PreparedText:
    """One extractor's text for a paper, in every form the matcher needs."""

    def __init__(self, raw_text: str):
        self.display, self.search = prepare_text(raw_text)
        self.key, self.positions = loose_key(self.search)


# --------------------------------------------------------------------------
# Matching
# --------------------------------------------------------------------------

def find_spans(text: PreparedText, excerpt: str):
    """Find every place the excerpt occurs. Returns (spans, method)."""
    target = clean_text(excerpt).lower()
    if not target:
        return [], None

    # Whole words only, so "SES" does not match inside "processes".
    pattern = r"(?<!\w)" + re.escape(target) + r"(?!\w)"
    spans = [m.span() for m in re.finditer(pattern, text.search)]
    if spans:
        return spans, "exact"

    target_key, _ = loose_key(target)
    if len(target_key) < MIN_LOOSE_LENGTH:
        return [], None

    position = text.key.find(target_key)
    while position != -1:
        last = position + len(target_key) - 1
        spans.append((text.positions[position], text.positions[last] + 1))
        position = text.key.find(target_key, position + 1)
    return spans, ("loose" if spans else None)


def context_window(text: PreparedText, span, window: int = CONTEXT_WINDOW):
    start = max(0, span[0] - window)
    end = min(len(text.display), span[1] + window)
    return text.display[start:end]


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
    # macOS stores accents as separate characters ("e" + accent) while the CSV
    # has them combined ("é"); split them so both end up as plain "e".
    name = unicodedata.normalize("NFKD", name)
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
    """Lazily extracts and caches a paper's text for each extractor.

    With a cache_dir, the raw extracted text is also saved to disk, so later
    runs skip the slow PDF extraction.
    """

    def __init__(self, pdf_path: Path, cache_dir: Path = None):
        self.pdf_path = pdf_path
        self.cache_dir = cache_dir
        self._cache = {}

    def _extract(self, name, fn):
        if self.cache_dir is None:
            return fn(self.pdf_path)

        stat = self.pdf_path.stat()
        cache_id = hashlib.sha256(
            f"{self.pdf_path}|{stat.st_size}|{stat.st_mtime_ns}".encode("utf-8")
        ).hexdigest()
        cached = self.cache_dir / f"{cache_id}.{name}.txt"
        if cached.exists():
            return cached.read_text(encoding="utf-8")

        raw_text = fn(self.pdf_path)
        temporary = cached.with_suffix(".tmp")
        temporary.write_text(raw_text, encoding="utf-8")
        temporary.replace(cached)
        return raw_text

    def get(self, name, fn):
        if name not in self._cache:
            try:
                self._cache[name] = PreparedText(self._extract(name, fn))
            except Exception as e:
                print(f"  WARNING: {name} extractor failed on "
                      f"{self.pdf_path.name}: {e}")
                self._cache[name] = PreparedText("")
        return self._cache[name]

    def texts(self):
        """Yield (extractor_name, PreparedText), skipping empty extractions."""
        for name, fn in EXTRACTORS:
            text = self.get(name, fn)
            if text.search:
                yield name, text


def no_match(status):
    return {"match_status": status, "context": None, "extractor": None,
            "method": None, "n_occurrences": 0, "score": None,
            "resolved_by": None}


def find_unique(paper_text: PaperText, excerpt: str):
    """Try each extractor in order; first one with exactly one hit wins.

    Returns (unique, ambiguous): each is (extractor, text, spans, method)
    or None.
    """
    ambiguous = None
    for name, text in paper_text.texts():
        spans, method = find_spans(text, excerpt)
        if len(spans) == 1:
            return (name, text, spans, method), None
        if spans and ambiguous is None:
            ambiguous = (name, text, spans, method)
    return None, ambiguous


def match_annotation(paper_text: PaperText, excerpt: str):
    """Locate one excerpt in the paper. Returns a dict of match fields."""
    unique, ambiguous = find_unique(paper_text, excerpt)

    if unique:
        name, text, spans, method = unique
        return {"match_status": "matched", "context": context_window(text, spans[0]),
                "extractor": name, "method": method, "n_occurrences": 1,
                "score": None, "resolved_by": None, "_span": spans[0]}

    if ambiguous:
        name, text, spans, method = ambiguous
        return {"match_status": "ambiguous_match",
                "context": context_window(text, spans[0]),
                "extractor": name, "method": method,
                "n_occurrences": len(spans), "score": None,
                "resolved_by": "first_occurrence"}

    # Excerpts highlighted in several pieces: match the longest piece we can.
    fragments = [f for f in excerpt.splitlines()
                 if len(loose_key(clean_text(f).lower())[0]) >= MIN_FRAGMENT_LENGTH]
    if len(fragments) > 1:
        for fragment in sorted(fragments, key=len, reverse=True):
            unique, _ = find_unique(paper_text, fragment)
            if unique:
                name, text, spans, _method = unique
                return {"match_status": "matched_partial",
                        "context": context_window(text, spans[0]),
                        "extractor": name, "method": "fragment",
                        "n_occurrences": 1, "score": None,
                        "resolved_by": None, "_span": spans[0]}

    # Last resort: closest approximate match across all extractors.
    target = clean_text(excerpt).lower()
    if len(target) >= MIN_FUZZY_LENGTH:
        best = None
        for name, text in paper_text.texts():
            alignment = fuzz.partial_ratio_alignment(
                target, text.search, score_cutoff=FUZZY_THRESHOLD)
            if alignment and (best is None or alignment.score > best[0]):
                best = (alignment.score, name, text,
                        (alignment.dest_start, alignment.dest_end))
        if best:
            score, name, text, span = best
            return {"match_status": "matched_fuzzy",
                    "context": context_window(text, span),
                    "extractor": name, "method": "fuzzy", "n_occurrences": 1,
                    "score": round(score, 1), "resolved_by": None, "_span": span}

    return no_match("excerpt_not_found")


def span_distance(a, b):
    """Characters between two spans (0 if they overlap)."""
    return max(a[0] - b[1], b[0] - a[1], 0)


def resolve_ambiguous(paper_text: PaperText, pair: dict, located: list):
    """Choose which occurrence of a repeated excerpt to use for its context.

    `located` holds the paper's other pairs that were found in one place.
    Updates `pair` in place.
    """
    paired_code = PAIRED_CODE.get(pair["code"])

    # Positions only compare within one extractor's text.
    candidates = []
    for name, text in paper_text.texts():
        spans, _method = find_spans(text, pair["annotation"])
        anchors = [a for a in located if a["extractor"] == name]
        if spans:
            candidates.append((name, text, spans, anchors))

    def choose(name, text, span, status, resolved_by):
        pair.update({"match_status": status, "extractor": name,
                     "context": context_window(text, span),
                     "resolved_by": resolved_by})

    # 1. Inside another located excerpt (the paired Supporting Info first).
    for paired_only in (True, False):
        for name, text, spans, anchors in candidates:
            for anchor in anchors:
                if paired_only != (anchor["code"] == paired_code):
                    continue
                start, end = anchor["_span"]
                inside = [s for s in spans if start <= s[0] and s[1] <= end]
                if inside:
                    choose(name, text, inside[0], "matched_anchored", "inside_excerpt")
                    return

    # 2. Closest to the paired Supporting Info excerpt.
    best = None
    for name, text, spans, anchors in candidates:
        for anchor in anchors:
            if anchor["code"] != paired_code:
                continue
            for span in spans:
                distance = span_distance(span, anchor["_span"])
                if distance <= NEAR_ANCHOR_DISTANCE and (best is None or distance < best[0]):
                    best = (distance, name, text, span)
    if best:
        _distance, name, text, span = best
        choose(name, text, span, "matched_anchored", "near_supporting_info")
        return

    # 3. Surrounded by words typical for this code. Still only a guess.
    cues = CUE_WORDS.get(pair["code"], [])
    if cues and candidates:
        name, text, spans, _anchors = candidates[0]
        scores = []
        for span in spans:
            nearby = text.search[max(0, span[0] - CUE_WINDOW):span[1] + CUE_WINDOW]
            scores.append(sum(cue in nearby for cue in cues))
        if max(scores) > 0:
            choose(name, text, spans[scores.index(max(scores))],
                   "ambiguous_match", "cue_words")


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
    parser.add_argument("--rerun", action="store_true",
                        help="redo papers that already have a checkpoint")
    args = parser.parse_args()

    papers = load_annotations(args.csv)
    pdf_index = build_pdf_index(args.pdf_dir)

    # One checkpoint file per publication, so a crash only loses one paper.
    checkpoint_dir = Path(args.output + ".checkpoints")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Extracted text is kept on disk, so reruns only redo the (fast) matching.
    text_cache_dir = Path(args.output + ".text_cache")
    text_cache_dir.mkdir(parents=True, exist_ok=True)

    for i, (paper_id, annotations) in enumerate(papers.items(), 1):
        checkpoint_id = hashlib.sha256(paper_id.encode("utf-8")).hexdigest()
        checkpoint = checkpoint_dir / f"{checkpoint_id}.jsonl"

        if checkpoint.exists() and not args.rerun:
            print(f"[{i}/{len(papers)}] Already processed: {paper_id}")
            continue

        print(f"[{i}/{len(papers)}] Processing: {paper_id}")

        pdf_path = find_pdf_for_paper(paper_id, pdf_index)
        paper_text = PaperText(pdf_path, text_cache_dir) if pdf_path else None

        paper_pairs = []

        for annotation in annotations:
            excerpt = annotation["content"]

            if paper_text is None:
                match = no_match("pdf_not_found")
            else:
                match = match_annotation(paper_text, excerpt)

            if match["context"] is None:
                match["context"] = excerpt  # fallback: annotator's excerpt alone

            paper_pairs.append({
                "source": paper_id,
                "annotator": annotation["annotator"],
                "code": annotation["code"],
                "annotation": excerpt,
                **match,
            })

        # Second pass: excerpts found in one place tell us where the repeated
        # ones (short labels like "SES") belong.
        located = [pair for pair in paper_pairs if "_span" in pair]
        for pair in paper_pairs:
            if pair["match_status"] == "ambiguous_match":
                resolve_ambiguous(paper_text, pair, located)
        for pair in paper_pairs:
            pair.pop("_span", None)

        # Write to a temp file, then rename, so a checkpoint is never half-written.
        temporary = checkpoint.with_suffix(".tmp")
        with open(temporary, "w", encoding="utf-8") as f:
            for pair in paper_pairs:
                f.write(json.dumps(pair, ensure_ascii=False) + "\n")
        temporary.replace(checkpoint)

    # Combine all completed publications into the final JSONL.
    status_counts = Counter()
    extractor_counts = Counter()
    method_counts = Counter()
    resolved_counts = Counter()
    stale_rows = 0

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
                    pair = json.loads(line)
                    status_counts[pair["match_status"]] += 1
                    if "resolved_by" not in pair:
                        stale_rows += 1
                    if pair.get("resolved_by"):
                        resolved_counts[pair["resolved_by"]] += 1
                    if pair.get("extractor"):
                        extractor_counts[pair["extractor"]] += 1
                    if pair.get("method"):
                        method_counts[pair["method"]] += 1

    total = sum(status_counts.values())
    print("\n=== SUMMARY (whole output file) ===")
    print(f"Publications in dataset: {len(papers)}")
    print(f"Annotations: {total}")
    for status in ["matched", "matched_partial", "matched_fuzzy",
                   "matched_anchored", "ambiguous_match",
                   "excerpt_not_found", "pdf_not_found"]:
        share = 100 * status_counts[status] / total if total else 0
        print(f"{status}: {status_counts[status]} ({share:.1f}%)")
    print("Matches by extractor:")
    for name, _ in EXTRACTORS:
        print(f"  {name}: {extractor_counts[name]}")
    print("Matches by method:")
    for method in ["exact", "loose", "fragment", "fuzzy"]:
        print(f"  {method}: {method_counts[method]}")
    print("Excerpts found more than once, resolved by:")
    for resolved_by in ["inside_excerpt", "near_supporting_info", "cue_words",
                        "first_occurrence"]:
        print(f"  {resolved_by}: {resolved_counts[resolved_by]}")
    if stale_rows:
        print(f"WARNING: {stale_rows} rows come from checkpoints written by an "
              f"older version of this script. Use --rerun to redo them.")
    print(f"Output: {output}")
    print(f"Checkpoints: {checkpoint_dir}")
    print(f"Text cache: {text_cache_dir}")


if __name__ == "__main__":
    main()