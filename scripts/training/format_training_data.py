"""
format_training_data.py
----------------------
This script is to organize matched annotations -> training examples.
match_pdfs_to_annotations.py gives one row per annotation (its code, its text,
and the real paragraph around it from the paper).
But a model learns from flat pairs: "here's a question (paper excerpt), here's
the correct answer (IV, DV, covariates, additional variables, and the
supporting info for the IV and DV)."
So for every excerpt we list the paper's labels that really appear in that
excerpt, one line per kind of label ("none" when the excerpt has none), and
write one prompt -> completion pair that finetune_braingpt.py can read.

USAGE
-----
    python3 scripts/training/format_training_data.py \
        --input training_pairs_with_context_v2.jsonl \
        --output data/processed/training/training_pairs_v3.jsonl
"""

import argparse
import json
import re
from collections import Counter, defaultdict

INSTRUCTION = ("Extract the independent variable(s) (IV), dependent variable(s) (DV), "
               "covariates, additional variables, and the supporting information for the "
               "IV and DV from this excerpt of a research paper.")

#annotation code -> (name of its line in the answer, longest label we keep)
#longer short-labels are whole sentences, not variable names; supporting info
#is a sentence or two by nature, so it gets more room
ANSWER_FIELDS = {
    "Independent Variable": ("IV", 150),
    "Dependent Variable": ("DV", 150),
    "Covariate": ("Covariates", 150),
    "Additional Variable": ("Additional variables", 150),
    "IV Supporting Info": ("IV supporting info", 600),
    "DV Supporting Info": ("DV supporting info", 600),
}

#rows whose context is a real, located paragraph of the paper
RELIABLE_STATUSES = {"matched", "matched_anchored", "matched_fuzzy", "matched_partial"}

#codes whose excerpt is likely to talk about the variables
EXCERPT_CODES = set(ANSWER_FIELDS) | {"Findings"}


def clean(text):
    return " ".join(text.split())


#keep only the labels that are written in the excerpt, as whole words
def labels_in_excerpt(labels, excerpt_lower):
    found = []
    for label in labels:
        pattern = r"(?<!\w)" + re.escape(label.lower()) + r"(?!\w)"
        if re.search(pattern, excerpt_lower):
            found.append(label)
    return found


def build_prompt(excerpt):
    return f'{INSTRUCTION}\n\nExcerpt: "{excerpt}"'


#one line per field, "none" when the excerpt has no label of that kind
def format_answer(found):
    lines = []
    for code, (name, _max_length) in ANSWER_FIELDS.items():
        lines.append(f"{name}: {'; '.join(found[code]) if found[code] else 'none'}")
    return "\n".join(lines)


def paper_labels(rows, code):
    max_length = ANSWER_FIELDS[code][1]
    labels = []
    for row in rows:
        label = clean(row["annotation"])
        if row["code"] == code and label and len(label) <= max_length \
                and label not in labels:
            labels.append(label)
    return labels


def build_examples_for_paper(rows, statuses, min_fields):
    examples = []
    labels = {code: paper_labels(rows, code) for code in ANSWER_FIELDS}
    seen_answers = set()

    for row in rows:
        if row["code"] not in EXCERPT_CODES or row["match_status"] not in statuses:
            continue

        excerpt = clean(row["context"])
        found = {code: labels_in_excerpt(labels[code], excerpt.lower())
                 for code in ANSWER_FIELDS}

        #an excerpt with nothing labelled in it teaches the model nothing
        if sum(1 for code in found if found[code]) < min_fields:
            continue

        #nearby annotations give almost the same excerpt; keep one per answer
        answer = format_answer(found)
        if answer in seen_answers:
            continue
        seen_answers.add(answer)

        examples.append({"prompt": build_prompt(excerpt), "completion": answer})
    return examples


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Matched annotations -> prompt/completion pairs")
    parser.add_argument("--input", default="training_pairs_with_context_v2.jsonl")
    parser.add_argument("--output", default="data/processed/training/training_pairs_v3.jsonl")
    parser.add_argument("--min-fields", type=int, default=1,
                        help="skip excerpts with fewer than this many kinds of label in them")
    parser.add_argument("--include-ambiguous", action="store_true",
                        help="also use excerpts whose location in the paper is a best guess")
    args = parser.parse_args()

    statuses = set(RELIABLE_STATUSES)
    if args.include_ambiguous:
        statuses.add("ambiguous_match")

    papers = defaultdict(list)
    status_counts = Counter()
    with open(args.input, "r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            papers[row["source"]].append(row)
            status_counts[row["match_status"]] += 1

    all_examples = []
    papers_with_examples = 0
    for rows in papers.values():
        examples = build_examples_for_paper(rows, statuses, args.min_fields)
        all_examples.extend(examples) #not list as we dont want nested loop
        if examples:
            papers_with_examples += 1

    with open(args.output, "w", encoding="utf-8") as f:
        for example in all_examples:
            f.write(json.dumps(example, ensure_ascii=False) + "\n")

    used = sum(status_counts[s] for s in statuses)
    print(f"Read {sum(status_counts.values())} annotations from {len(papers)} papers "
          f"({used} with a usable location)")
    print(f"Wrote {len(all_examples)} training examples from {papers_with_examples} papers")
    for _code, (name, _max_length) in ANSWER_FIELDS.items():
        filled = sum(1 for e in all_examples if f"\n{name}: none" not in "\n" + e["completion"])
        print(f"  with {name}: {filled}")
    print(f"Saved to {args.output}")
