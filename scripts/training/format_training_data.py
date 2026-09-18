"""
format_training_data.py
----------------------
This script is to organize JSON -> training examples. 
So the problem is BrainGPT can't learn anything from synthetic_annotations.json in its current shape 
That file is organized for humans to read easily — nested, labeled, clean.
But a model learns from flat pairs: "here's a question (paper text), here's the correct answer (extracted variable info)." 
So it will flatten it into big list of these question -> answer pairs, one pair per IV/DV relationship --> basically so BrainGPT can read it 
"""

import csv 
import sys
import json 
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_PATH = os.path.join(SCRIPT_DIR, "output", "synthetic_annotations_large.json")
OUTPUT_PATH = os.path.join(SCRIPT_DIR, "output", "training_examples.json")
#pull out IV/DV text, turn the covariates text into readable comma-seperated string -> glue it all into one sentence and hands its back
def format_answer(variables):
    iv = variables["independent_variable"]
    dv = variables["dependent_variable"]
    
    covariances = variables.get("covariances", [])
    covariances_text = ", ".join(covariances) if covariances else "none reported"
    
    answer = f"Independent variable: {iv}. Dependent variable: {dv}. Covariates: {covariances_text}."
    return answer 

#take the paper source_text and wraps it in an instruction, os the model
#know where is the question 

def build_prompt(source_text):
    instruction = "Extract the independent variable, dependent variable, covariances from this text"
    prompt = f"{instruction}\n\nText: {source_text}"
    return prompt
def build_examples_for_paper(paper):
    examples = []
    prompt = build_prompt(paper["source_text"])
    
    if paper["variables"]:
        for variable in paper["variables"]:
            answer = format_answer(variable)
            examples.append({"prompt": prompt, "completion": answer})
    else:
        answer = "No independent/dependent variable relationships was reported in this paper"
        examples.append({"prompt": prompt, "completion": answer})
    return examples

if __name__ == "__main__":
    with open(INPUT_PATH, "r", encoding="utf-8") as f:
        papers = json.load(f)
        
    all_examples = []
    for paper in papers:
        examples = build_examples_for_paper(paper)
        all_examples.extend(examples) #not list as we dont want nested loop
        
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(all_examples, f, indent=2)
        
    print(f"Processed {len(papers)} papers -> {len(all_examples)} training examples")
    print(f"Saved to training_examples.json")