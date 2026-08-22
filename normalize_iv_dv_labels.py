"""
normalize_iv_dv_labels.py
--------------------------

                    ┌─────────────────┐
                    │ annotations.csv │
                    └────────┬────────┘
                             │
                             ▼
                        read rows
                             │
                             ▼
                     extract raw labels
                             │
                             ▼
                        deduplicate
                             │
                             ▼
                  ["BMI", "screen time", ...]
                             │
                             ▼
                         batching
                             │
                             ▼
                       build_prompt()
                             │
                             ▼
                       call_claude()
                             │
                             ▼
                canonical labels returned
                             │
                             ▼
                    create dictionary
                             │
             ┌───────────────┴───────────────┐
             │                               │
             ▼                               ▼
      "BMI" → "body mass index"       sanity checking
             │
             ▼
       apply mapping to rows
             │
             ▼
      normalized_labels.csv
      
Maps raw IV/DV label text (often full sentence fragments from Atlas.ti annotations) 
to short, canonical labels, so the same construct expressed different ways accross 
papers (e.g. "BMI" vs "body mass index) collapses into ONE node in the Task 2 knowledge graph 
instead of many.

Pipeline:
1. Load a CSV of annotations. Must contain a column with raw IV/DV text.
2. Deduplicate raw label strings (many repeat across papers).
3. Batch the unique labels (default 20/call) into a prompt with a few-shot
examples of what a "clean" canonical label looks like.
4. Call the OpenAI API to get canonical labels back as JSON. 
5. Parse the response into a {raw_label: normalized_label} dict.
6. Apply that mapping to every row in the original CSV, adding a new
"normliazed_label" column.
7. Sanity-check: flag any normalized label that's still long/sentence-like {word count over threshold}
for manual review, since that usually means the mode failed to compress it

Usage:
    python normalize_iv_dv_labels.py
       --input annotations.csv \
       --label--column raw_label \
       --output normalized_labels.csv

Requires:
    pip install anthropic --break-system-packages
    export ANTHROPIC_API_KEY=sk-ant-...
"""

import csv
import json 
import os #make the python interact with operating system
import sys #stop itself from sth wrong
import time
import re #regular expressions
import argparse

import anthropic 

#note DictReader
MODEL = "claude-sonnet-4-5"
BATCH_SIZE = 20 
SLEEP_BETWEEN_CALLS = 1.0
MAX_RETRIES = 3

FEW_SHOT_EXAMPLES = """\
Raw: "the seven indices of adversity"
Canonical: adversity

Raw: "parents' familism values"
Canonical: familism values

Raw: "functional and structural connectivity"
Canonical: brain connectivity
 
Raw: "gray matter volume (GMV) of cortical regions of interest (ROIs)"
Canonical: gray matter volume
 
Raw: "BMI"
Canonical: body mass index
 
Raw: "time spent on electronic devices"
Canonical: screen time
 
Raw: "age"
Canonical: age 
"""

def build_prompt(raw_labels, existing_canonicals):
    """Build one prompt asking Claude to normalize a batch of raw labels.
    So its create the empty list, and then number every new raw label, and join them with newlines with put the numbered labels into the prompt
    """
    numbered_labels = []
    
    for i, label in enumerate(raw_labels, start=1):
        line = f"{i}. {label}" 
        numbered_labels.append(line)
        
    numbered = "\n".join(numbered_labels)
    
    canonical_lines = []
    for canonical in existing_canonicals:
        canonical_lines.append(f"- {canonical}")
        
    canonical_text = "\n".join(canonical_lines)
    
    return f"""You are normalizing variable labels extracted from ABCD Study research papers, 
\ so the same underlying construct is labeled identically across papers even when authors \
    phrase its differently (abbreviations, synonyms, sentence fragments).
    
Rules:
- Canonical label should be {MAX_WORDS_CLEAN} words or fewer,
- Use most common and standard research term for the construct, not the abbreviation, 
\
    unless the abbreviation IS the standard term (e.g, IQ).
-Two raw labels that refers to the SAME construct must map to the EXACT same canonical string.
- Do not merge labels that are genuinely different constructs, even if related \
     (e.g, "screen time and "social media use" are TOTALLY DIFFERENT)
     
Examples:
{FEW_SHOT_EXAMPLES}

Existing canonical labels:
{canonical_text}
 
Now normalize these {len(raw_labels)} raw labels:
{numbered}

Respond with ONLY a JSON Array of {len(raw_labels)} strings (the canonical labels, in the \
same order as the input, no numbering, no preamble, no markdown fences):"""


#-------API CLAUDE/OPENAI KEY API------------
"""
Sometime its would failed because the network issues, rate limit
temporary Claude error, invalid JSON 
"""

def call_claude(client, raw_labels, existing_canonicals):
    """
    client
    → connection to Claude

    raw_labels
    → labels we're trying to normalize now

    existing_canonicals
    → identities we've already established
    """
    prompt = build_prompt(raw_labels, existing_canonicals)
    
    response = client.messages.create(
        model=MODEL, 
        max_tokens=2000,
        messages=[
            {
                "role":"user", 
                "content":prompt
            }
        ],
    )
    
    text = response.content[0].text.strip()
    
    parsed = json.load(text)
    return parsed 

#===========NORMALIZED THE LABELS===========
"""
take all unique raw labels
        ↓
split them into batches
        ↓
send each batch to Claude
        ↓
keep track of previous canonical labels
        ↓
build this final mapping:

{
    "BMI": "body mass index",
    "BMI score": "body mass index",
    "screen usage": "screen time"
}
"""
def normalize_labels(unique_labels, client): #unique_labels is the deduplicate raw labels from ur csv 
    mapping = {} #dictionary 
    existing_canonicals = set()
    
    batches = []
    #so after we have all the empty list we need to add it into batch first 
    #then we will call claude to match the raw labels with the existing canonical 
    for i in range(0, len(unique_labels), BATCH_SIZE):
        batch = unique_labels[i:i + BATCH_SIZE]
        batches.append(batch)
        
    for batch in batches: #loop thru all the canonical we have
        canonical_labels = call_claude(
            client, 
            batch, 
            existing_canonicals #its update after every batch
        )
        
        if len(canonical_labels) != len(batch):
            print("Warning: Claude returned the wrong number of labels")
            continue
        for raw, canonical in zip(batch, canonical_labels):
            mapping[raw] = canonical 
        for canonical in canonical_labels: #update the existing canonical list
            existing_canonicals.add(canonical)
            
    return mapping 
#=======SANITY CHECK========
"""
Does this output look like a real canonical construct, or does it look like
Claude accidentally returned a sentence fragment/explanation/malformed answer?

Will FLAG if these happens:
- Claude/OpenAI return empty label
- The canonical is too long, like >12 words
- The canonical still look like a sentence, e.g. contains phrases like "participants reported" or "was associated with"
- Claude leaves a long sentence like raw label complete unchanged 
- The canonical contains a newline or weird formatting

EXAMPLE: 
    resting-state functional connectivity of the default mode network
--> This could be a legitmate construct even its long
"""
def sanity_check(mapping):
    flagged = {} #to store flagged labels
    
    for raw, canonical in mapping.items()):
        reasons = []
        if not canonical.strip(): #if theres nothing meaningful in the canonical label, flag it
            reasons.append("empty canonical label")
        word_count = len(canonical.strip()):
        if word_count > 12:
            reasons.append("canonical label is unusually long")
            
        sentence_markers = [
            "was associated with",
            "were associated with",
            "is associated with",
            "participants reported",
            "children reported",
            "adolescents reported",
            "the relationship between",
            "effect of",
            "effects of"
        ]
        canonical_lower = canonical.lower()
        for marker in sentence_markers:
            if marker in canonical_lower:
                reasons.append(f"looks sentence-like: contains '{marker}")
                break 
        raw_word_count = len(raw.split())
        
        if row.lower().strip() == canonical.lower().strip() and raw_word_count >8:
            reasons.append("long raw label was left unchanged")         
        if "\n" in canonical:
            reasons.append("canonical label contains a newline")
            
        if canonical.startswith("- ") or canonical.startswith("* "):
            reasons.append("canonical label contains list formatting")
        
        if reasons:
            flagged[raw] = {
                "canonical": canonical, 
                "reasons": reasons
            }           
    if flagged:
        print(f"\nSanity check flagged {len(flagged)} mapping for review:")
        for raw, info in flagged.items():
            print(f"\nRaw: {raw}")
            print(f"Canonical: {info['canonical']}")
            for reason in info["reasons"]:
                print(f" - {reason}")
    return flagged
#==========SECOND SANITY CHECK==========
"""
After first sanity check:
{
    "BMI": "body mass index",
    "BMI score": "body mass index",
    "adolescent BMI": "BMI measure",
    "screen usage": "screen time",
    "electronic device use": "screen time"
}
PROBLEM is: 
body mass index
BMI measure
Ask Claude: Does each canonical match one fo the canonical identities we've already accepted
"""
def check_canonical_duplicates(mapping, client):
    canonical_labels = sorted(set(mapping.values())) #get the value from dict
    canonical_mapping = {}
    master_canonicals = set()
    batches = []
    
    for batch in (0, len(canonical_labels), BATCH_SIZE):
        batch = canonical_labels[i:i + BATCH_SIZE]
        batches.append(batch) #process the canonical labels in batches
        resolved = call_claude(client, batch, master_canonicals)
        
        for old_canonical, final_canonical in zip(batch, resolved):
            canonical_mapping[old_canonical] = final_canonical
        for final_canonical in resolved:
            master_canonicals.add(final_canonical)
    #update the original raw -> canonical mapping            
    final_mapping = {}
    for raw, old_canonical in mapping.items():
        final_mapping[raw] = canonical_mapping.get(old_canonical, old_canonical)
    return final_mapping
"""
{
    "BMI": "body mass index",
    "BMI score": "body mass index",
    "adolescent BMI": "body mass index",
    "device usage": "screen time",
    "screen exposure": "screen time"
}
So after the second sanity check, we have final mapping like this
BMI ──────────────┐
BMI score ────────┼──→ body mass index
adolescent BMI ───┘
"""



#==========CONNECT ALL THE FUNCTION TOGETHER=========
def main():
    parser = argparse.ArgumentParser(
        description="Normalize raw IV/DV labels using Claude/OpenAI"
        
    )
    parser.add_argument("--input", required=True, help="Path to input CSV")
    parser.add_argument("--label_column", required=True, help="Column containing the raw IV/DV lables")
    parser.add_argument("--output", required=True, help="Path for output CSV")
    args = parser.parse_args()
    
    if not os.environ.get("ANTHROPIC_API_KEY"):
        sys.exit(
            "ERROR":ANTHROPIC_API_KEY is not set.""
        )
    with open(args.input, 
              newline="", 
              encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        fieldnames = reader.fieldnames
    if args.label_column not in fieldnames:
        sys.exit(
            f"ERROR": column {args.label_column!r} not found" #!r help with debugging
            f"Available columns: {fieldnames}"
        )
        
    raw_values = []
    for row in rows:
        raw = rows[args.label_column].strip()
        
        if raw:
            raw_values.append(row)
    unique_labels = sorted(len(raw_values))
    print(
        f"Loaded {len(rows)} rows"
        f"with {len(unique_labels)} unique raw labels"
    )
    client = anthropic.Anthropic()
    mapping = normalize_labels(
        unique_labels, 
        client
    )
    mapping = deduplicate_canonicals(
        mapping, 
        client
    )
    flagged = sanity_check(mapping)
    print(
        f"{len(flagged)} mappings flagged for manual review"
    )
    #prepare the output columns
    out_fieldnames = fieldnames + ["normalized_label"]
    with open(
        args.output, 
        "w",
        newline="",
        encoding="utf-8"
    ) as f:
    
    writer = csv.DictWriter(
        f,
        fieldnames=out_fieldnames
    )
    
    write.writeheader()
    for row in rows:
        raw = rows[args.label_column].strip()
        
        row["normalize_label"] = mapping.get(raw, raw)
        writer.writerow(row)
        
    final_canonicals = set(mapping.values())
    print(f"\nDone.")
    print(f"Wrote {len(rows)} rows to {args.output}")
    print(
        f"{len(unique_labels)} unique raw labels "
        f"collapsed into "
        f"{len(final_canonicals)} canonical labels."
    )
if __name__ == "__main__":
    main