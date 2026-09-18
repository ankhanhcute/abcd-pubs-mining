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
    pip install openai --break-system-packages
    export OPENAI_KEY_API=sk-ant-...
"""

"""
Update on the pipeline now, we need the scientific rules, so right now the model return one string. 
So its cannot represent one annotation --> TWO construct 
So instead I will change it to:
[
  {
    "status": "valid",
    "canonicals": ["parental acceptance and warmth"]
  },
  {
    "status": "multiple",
    "canonicals": [
      "nucleus accumbens fractional anisotropy",
      "body mass index"
    ]
  },
  {
    "status": "multiple",
    "canonicals": [
      "air pollution exposure",
      "emotional problems"
    ]
  },
  {
    "status": "exclude",
    "canonicals": []
  }
]

"""

import csv
import json 
import os #make the python interact with operating system
import sys #stop itself from sth wrong
import time
import re #regular expressions
import argparse

from openai import OpenAI

#note DictReader
MODEL = "gpt-5.4"
BATCH_SIZE = 50 
CACHE_FILE = "normalization_cache.json"

FEW_SHOT_EXAMPLES = """\
Raw: "maternal acceptance"
Output:
{"status": "valid",
 "canonicals": ["parental acceptance and warmth"]}

Raw: "The left NAcc fractional anisotropy and the BMI were our predictor and outcome"
Output:
{"status": "multiple",
 "canonicals": [
     "nucleus accumbens fractional anisotropy",
     "body mass index"
 ]}

Raw: "relationship between air pollution exposure and emotional problems"
Output:
{"status": "multiple",
 "canonicals": [
     "air pollution exposure",
     "emotional problems"
 ]}

Raw: "GLMMs included family unit and research site as random intercepts"
Output:
{"status": "exclude",
 "canonicals": []}
"""
#------CACHE TO SAVE THE SUCCESSFUL RESULT TO FILE------
def load_cache():
    if not os.path.exists(CACHE_FILE):
        return {}
    with open(CACHE_FILE, "r", encoding="utf-8") as f:
        return json.load(f)
    
def save_cache(cache):
    with open(CACHE_FILE, "w", encoding="utf-8") as f:
        json.dump(
            cache, 
            f, 
            indent=2, 
            ensure_ascii=False
        )
def build_prompt(raw_labels):
    """Build one prompt asking Claude to normalize a batch of raw labels.
    So its create the empty list, and then number every new raw label, and join them with newlines with put the numbered labels into the prompt
    """
    numbered_labels = []
    
    for i, label in enumerate(raw_labels, start=1):
        line = f"{i}. {label}" 
        numbered_labels.append(line)
        
    numbered = "\n".join(numbered_labels)
            
    return f"""You are normalizing IV/DV variable annotations extracted from ABCD Study research papers.

The goal is NOT simply to shorten or clean the text.

The goal is to identify the actual research construct or constructs represented by each annotation
so that equivalent constructs can share the same canonical identity in a knowledge graph.

For each raw annotation, classify it into ONE of these statuses:

1. "valid"
   - The annotation represents one research construct.
   - Return exactly one canonical construct.

2. "multiple"
   - The annotation clearly contains two or more distinct research constructs.
   - Return each construct separately.
   - NEVER combine separate constructs into one canonical label.

3. "exclude"
   - The annotation is methodological text, a statistical procedure, random-effect description,
     generic heading, or otherwise does not represent a meaningful IV/DV construct.
   - Return an empty canonicals list.

Normalization rules:

- Do not merge constructs merely because they are related.
- Preserve meaningful scientific distinctions.
- Use clear, standard research terminology when the annotation unambiguously refers to a known construct.
- Different wording or synonyms may be normalized to the same standard construct name, but preserve measurement-specific
distinctions when they materially change the scientific meaning. 
- Preserve distinction that materially change the scientific meaning of a construct. Do not merge 
constructs merely because they are related or belong to the same domain.
- If one annotation clearly refers to multiple distinct research construct
return each construct separately rather than combining them into one canonical label.
- Exclude text that describes statistical procedures, model specification, random effects, software, study 
design, generic headings, or other methodological information rather than actual independent or dependent variable.
- Do not create canonical nodes from meaningless fragments, section headings, 
explanatory prose, or methodological details. 
-  Canonical labels should use clear, standard research terminology while preserving 
scientifically meaningful information 
- Do not infer constructs that are not supported by the annotation text.
- Do not guess the meaning of an ambiguous abbreviation or acronym. Only expand an abbreviation when its meaning is explicitly stated in the 
annotation or can be determined with high confidence from the text itself. If the meaning is ambiguous, preserve the abbreviation rather than inventing an expansion. 

Examples: {FEW_SHOT_EXAMPLES}
 
Now normalize these {len(raw_labels)} raw labels:
{numbered}

Respond with ONLY a JSON array of exactly {len(raw_labels)} objects, in the same order as the input.

Each object must have exactly this structure:
{{
    "status": "valid" | "multiple" | "exclude"
    "canonicals": ["canonical construct", "..." ]
    
}}

Do not include numbering, explanations, markdown fences, or any text outside the JSON array.
"""

#-------API CLAUDE/OPENAI KEY API------------
"""
Sometime its would failed because the network issues, rate limit
temporary Claude error, invalid JSON 
"""

def call_claude(client, raw_labels):
    """
    client
    → connection to OpenAI

    raw_labels
    → labels we're trying to normalize now

    existing_canonicals
    → identities we've already established
    """
    prompt = build_prompt(raw_labels)
    
    response = client.responses.create(
        model=MODEL, 
        input=prompt
    )
    
    text = response.output_text.strip()
    
    parsed = json.loads(text) #loads read JSON from a string
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
    mapping = load_cache() #dictionary
    labels_to_process = []
    
    for label in unique_labels:
        if label not in mapping:
            labels_to_process.append(label) 
    print(
        f"Cached: {len(unique_labels) - len(labels_to_process)} | "
        f"Need API: {len(labels_to_process)}"
    )


    batches = []
    #so after we have all the empty list we need to add it into batch first 
    #then we will call claude to match the raw labels with the existing canonical 
    for i in range(0, len(labels_to_process), BATCH_SIZE):
        batch = labels_to_process[i:i + BATCH_SIZE]
        batches.append(batch)
        
    for batch_num, batch in enumerate(batches, start=1):
        print(
            f"Batch {batch_num}/{len(batches)} | "
        f"new labels: {len(batch)} | ")
        canonical_labels = call_claude(
            client, 
            batch,
        )
        
        if len(canonical_labels) != len(batch):
            print("Warning: OpenAI returned the wrong number of labels")
            continue
        for raw, results in zip(batch, canonical_labels):
            mapping[raw] = results  #is already one dict
        save_cache(mapping)
            
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
    
    for raw, result in mapping.items():
        reasons = [] #need to be inside the loop, 
        #so its mean reasons from one raw label can carry over into the next raw 
        status = result["status"]
        if status == "valid" and len(result["canonicals"]) != 1:
            reasons.append("The status said VALID but there was NOTHING in the string")
        elif status == "multiple" and len(result["canonicals"]) < 2:
            reasons.append("The status said MULTIPLE but there was less than 2 canonicals")
        elif status == "exclude" and len(result["canonicals"]) != 0:
            reasons.append("EXCLUDE status should contain no canonicals")
        for canonical in result["canonicals"]:
            if not canonical.strip(): #if theres nothing meaningful in the canonical label, flag it
                reasons.append("empty canonical label")
            word_count = len(canonical.split())
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
        
            if raw.lower().strip() == canonical.lower().strip() and raw_word_count >8:
                reasons.append("long raw label was left unchanged")         
            if "\n" in canonical:
                reasons.append("canonical label contains a newline")
            
            if canonical.startswith("- ") or canonical.startswith("* "):
                reasons.append("canonical label contains list formatting")
        
        if reasons:
            flagged[raw] = {
            "status": status,
            "canonical": result["canonicals"], 
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
Ask OpenAI: Does each canonical match one fo the canonical identities we've already accepted
"""
def build_dedup_prompt(canonical_labels, master_canonicals):
    numbered_labels = []
    for i, label in enumerate(canonical_labels, start=1):
        numbered_labels.append(f"{i}. {label}")
        
    numbered = "\n".join(numbered_labels)
    
    master_lines = []
    
    for canonical in sorted(master_canonicals):
        master_lines.append(f"- {canonical}")
    
    if master_lines:
        master_text = "\n".join(master_lines)
    else:
        master_text = "None yet"
    return f"""
    You are deduplicating scientific construct labels. 
 
    Every input label already represents exactly ONE valid scientific. 
 
    Your task is ONLY to resolve duplicate construct identities. 
 
    Rules:
    - If an input label means the SAME underlying scientific construction as an 
    existing master canonical, return that EXACT master canonical string.
    - If two labels in the current batch mean the SAME underlying construct, 
    return the SAME canonical string for both. 
    - If no equivalent construct exists, keep the input label unchanged,
    - Do NOT exclude constructs.
    - Do NOT split constructs. 
    - Do NOT merge constructs merely because they are related 
    - Preserved scientifically meaningful distinctions. 
    - If an annotation contains methodological or statistical language but also explicitly identifies a meaningful 
    IV/DV construct, extract the construct rather than excluding the entire annotation.
Existing master canonicals:
{master_text}
Canonical candidates:
{numbered}
Return ONLY a JSON array of exactly {len(canonical_labels)} strings,
in the same order as the input.

Do not include explanations, numbering, or markdown.
    """
    
# ------- DEDUP CALL OPENAI API FOR THE SECOND PASS------
def call_dedup_openai(client, canonical_labels, master_canonicals):
    prompt = build_dedup_prompt(
        canonical_labels, 
        master_canonicals
    )
    
    response = client.responses.create(
        model=MODEL,
        input=prompt
    )
    
    text = response.output_text.strip()
    parsed = json.loads(text)
    return parsed
def check_canonical_duplicates(mapping, client):
    canonical_set = set()
    for result in mapping.values():
        for canonical in result["canonicals"]:
            canonical_set.add(canonical)
    canonical_labels = sorted(canonical_set)

    canonical_mapping = {}
    master_canonicals = set()
    batches = []
    
    for i in range(0, len(canonical_labels), BATCH_SIZE):
        batch = canonical_labels[i:i + BATCH_SIZE]
        batches.append(batch)
    for batch in batches:
        response = call_dedup_openai(client, batch, master_canonicals)
        if len(response) != len(batch):
            print("Warning: OpenAI returned wrong number of canonical labels")
            continue
        for old_canonical, final_canonical in zip(batch, response):
            final_canonical = final_canonical.strip()
            canonical_mapping[old_canonical] = final_canonical
            master_canonicals.add(final_canonical)    #update the original raw -> canonical mapping            
    final_mapping = {}
    for raw, result in mapping.items():
        final_canonicals = []
        for old_canonical in result["canonicals"]:
            resolved_canonical = canonical_mapping.get(
                old_canonical, 
                old_canonical
            )
            final_canonicals.append(resolved_canonical)
        final_mapping[raw] = {
        "status": result["status"],
        "canonicals": final_canonicals
        }
    return final_mapping
"""
"BMI"
→ {
    status: valid,
    canonicals: [body mass index]
}
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
    
    if not os.environ.get("OPENAI_API_KEY"):
        sys.exit(
            "ERROR:OPENAI_API_KEY is not set."
        )
    with open(args.input, 
              newline="", 
              encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        fieldnames = reader.fieldnames
    if args.label_column not in fieldnames:
        sys.exit(
            f"ERROR: column {args.label_column!r} not found" #!r help with debugging
            f"Available columns: {fieldnames}"
        )
        
    raw_values = []
    for row in rows:  #list can be indexed not string like content
        raw = row[args.label_column].strip()
        
        if raw:
            raw_values.append(raw)
        #row = whole csv row dictionary 
        #raw just the content text
    #deduplicate labels been eliminated
    unique_labels = sorted(set(raw_values)) #want the list of dedup not the len
    print(
        f"Loaded {len(rows)} rows"
        f"with {len(unique_labels)} unique raw labels"
    )
    #create the openai client
    client = OpenAI()
    #normalize the variable 
    first_mapping = normalize_labels(
        unique_labels, 
        client
    )
    #sanity the dedup canonical labels
    final_mapping = check_canonical_duplicates(
        first_mapping, 
        client
    )
    #flagged again
    flagged = sanity_check(first_mapping)
    print(
        f"{len(flagged)} mappings flagged for manual review"
    )
    #prepare the output columns
    out_fieldnames = fieldnames + ["normalized_status", "normalized_label"]
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
    
        writer.writeheader()
        for row in rows:
            raw = row[args.label_column].strip()
            result = final_mapping.get(raw) #just get the dict
            """
            Look inside the mapping dict using raw as the key, if its exits, give me the value, if doesnt, return 
            None
            """
            if result:
                row["normalized_status"] = result["status"]
                row["normalized_label"] = " | ".join(result["canonicals"])
            else:
                row["normalized_status"] = "N/A"
                row["normalized_label"] = raw
                
            writer.writerow(row)
            
    
    final_canonicals = set()
    for result in final_mapping.values():
        for canonical in result["canonicals"]:
            final_canonicals.add(canonical)
    
    print(f"\nDone.")
    print(f"Wrote {len(rows)} rows to {args.output}")
    print(
        f"{len(unique_labels)} unique raw labels "
        f"collapsed into "
        f"{len(final_canonicals)} canonical labels."
    )
if __name__ == "__main__":
    main()