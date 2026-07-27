"""
parser.py
-----------------
This code to normalize the raw annotation exports (CSV, one row per IV/DV pair)
onto the nested per-paper JSON schema used by the fine-tuning pipeline. 
There are 3 parts in the file:
1. A translation dictionary (which columns names mean what)
2. A machine that reads the raw file and organizes it (the main worker)
3. A checker thats look for the mistakes (quality control)
"""

import csv 
import json
import sys 
from collections import OrderedDict
#map from expected raw column name -> internal field name 
#update the left-hand side if the real Atlas.ti export use the diff headers]


COLUMN_MAPS = {
    "paper_id": "paper_id",
    "abcd_release": "abcd_release",
    "time_point": "time_point",
    "participant_criteria": "participant_criteria",
    "sample_size": "sample_size",
    "statistical_model": "statistical_model",
    "independent_variable": "independent_variable",
    "iv_supporting_info": "iv_supporting_info",
    "dependent_variable": "dependent_variable",
    "dv_supporting_info": "dv_supporting_info",
    "covariates": "covariates",
    "additional_variables": "additional_variables",
    "findings": "findings",
}

PAPER_LEVEL_FIELDS = [
    "abcd_release", "time_point", "participant_criteria",
    "participant_age", "sample_size", "statistical_model", "findings"
]

VARIABLE_LEVEL_FIELDS = [
    "independent_variable", "iv_supporting_info",
    "dependent_variable", "dv_supporting_info",
]

LIST_FIELDS = ["covariates", "additional_variables"]
#semicolon-seperated in the raw export 

def _split_list(value):
    if not value:
        return []
    return [v.strip() for v in value.split(";") if v.strip()]

def _clean(value):
    return value.strip() if value else None

def parse_csv(path):
    papers = OrderedDict()
    
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        missing_cols = [c for c in COLUMN_MAPS if c not in reader.fieldnames]
        if missing_cols:
            print(f"WARNING: expected columns not found in export: {missing_cols}", file=sys.stderr)
        
        for row in reader:
            paper_id = _clean(row.get("paper_id"))
            if not paper_id:
                print(f"WARNING: skipping row with no paper_id: {row}", file=sys.stderr)
                continue
            
            if paper_id not in papers:
                entry = {"paper_id": paper_id, "variables": []}
                for field in PAPER_LEVEL_FIELDS:
                    raw_val = _clean(row.get(field))
                    if field == "sample_size" and raw_val is not None:
                        try:
                            raw_val = int(raw_val)
                        except ValueError:
                            print(f"WARNING: skipping row with no paper_id: {row}", file=sys.stderr)
                    entry[field] = raw_val
                papers[paper_id] = entry 
                
            variable_entry = {}
            for field in VARIABLE_LEVEL_FIELDS:
                variable_entry[field] = _clean(row.get(field))
            for field in LIST_FIELDS:
                variable_entry[field] = _clean(row.get(field))
            
            papers[paper_id]["variables"].append(variable_entry)
            
    return list(papers.values())
def validate(papers):
    "basic sanity checks - flags issues so its not silently feeding"
    
    problems = []
    for p in papers:
        if not p.get("paper_id"):
            problems.append("missing_paper_id")
        if not p.get("variables"):
            problems.append(f"{p["paper_id"]}: no IV/DV available")
        
        for v in p.get("variables", []):
            if not v.get("independent_variable"):
                problems.append(f"{p['paper_id']}: variable entry missing independent_variable")
            if not v.get("dependent_variable"):
                problems.append(f"{p['paper_id']}: variable entry missing dependent_variable")
        if not p.get("findings"):
            problems.append(f"{p["paper_id"]}: missing finding text")
    return problems
"""
Example of one of the parser doing 
[
  {
    "paper_id": "P001",
    "variables": [
      {
        "independent_variable": "income",
        "dependent_variable": "depression"
      },
      {
        "independent_variable": "parent education",
        "dependent_variable": "working memory"
      }
    ]
  },
  {
    "paper_id": "P002",
    "variables": [
      {
        "independent_variable": "sleep",
        "dependent_variable": "anxiety"
      }
    ]
  }
]
"""
if __name__ == "__main__":
    input_path = sys.argv[1] if len(sys.argv) > 1 else "mock_raw_export.csv"
    output_path = sys.argv[2] if len(sys.argv) > 2 else "parsed_output.json"
     
    papers = parse_csv(input_path)
    problems = validate(papers)
    
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(papers, f, indent=2)
        
    print(f"Parsed {len(papers)} papers -> {output_path}")
    if problems:
        print(f"\n{len(problems)} validation warning(s):")
        
        for prob in problems:
            print(f"  - {prob}")
            
    else: 
        print("No validation issue found.")