import argparse 
import pandas as pd 
import numpy as np 
import os 
import sys 
from openai import OpenAI

VARIABLE_CODES = {
    "Independent Variable": "IV", 
    "Dependent Variable" : "DV",
    "Covariate": "Covariate",
    "Additional Variable":"Additional Variable",
     
}
EXPECTED_EMBEDDING_DIM = 1536 

def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Search the ABCD dictionary for many annotations labels."
    )
    parser.add_argument("--annotations", required=True, help="Path to be the annotation CSV file.")
    parser.add_argument("--limit", type=int, default=None, help="Optional number of unique labels to test.")
    parser.add_argument("--vectors", default="data/embeddings/dict_embeddings.npy",help="Path to the dictionary embedding vectors")
    parser.add_argument("--metadata", default="data/embeddings/dict_embeddings_meta.csv", help="Path to the metadata of the dictionary embeddings")
    parser.add_argument("--batch-size", type=int, default=100, help="Number of unique queries processed in one batch")
    return parser.parse_args()

def load_annotations(path):
    print(f"Reading annotations from: {path}")
    
    annotations = pd.read_csv(path)
    
    if "annotations_id" not in annotations.columns:
        annotations.insert(0, "annotations_id", range(len(annotations)))
    print(f"Rows: {len(annotations):,}")
    print(f"Columns: {list(annotations.columns)}")
    
    return annotations
def iter_query_batches(unique_queries, batch_size):
    if batch_size <= 0:
        raise ValueError(
            "Batch size must be greater than zero."
        )
    total_queries = len(unique_queries)
    #ex with 12 queries n batch size of 5 range(0, 12, 5)
    for start in range(0, total_queries, batch_size):
        end = min(start + batch_size, total_queries)
        batch = unique_queries.iloc[start:end].copy()
        yield start, end, batch #provide one batch at a time
def prepare_variable_annotations(annotations, limit=None):
    
    variable_annotations = annotations[annotations["Codes"].isin(VARIABLE_CODES)].copy()
    variable_annotations["variable_role"] = (
        variable_annotations["Codes"].map(VARIABLE_CODES)
    )
    print()
    print(f"Variable rows selected: {len(variable_annotations):,}")
    print(variable_annotations["Codes"].value_counts().to_string())
    print(variable_annotations[["annotations_id", "Codes", "variable_role", "Content"]].head().to_string(index=False))
    
    #query text
    variable_annotations["query_text"] = (
    variable_annotations["Content"]
    .fillna("")
    .astype(str)
    .str.replace(r"\s+", " ", regex=True)
    .str.strip(" ,;\t\r\n")
) 
    non_empty = (variable_annotations["query_text"] != "") #remove empty queries, remove rows contain nothing
    variable_annotations = variable_annotations[non_empty].copy()
    """
    So because the content have some problem with the \n or typo, so I create query text which 
    easier for model to read it, so it will change the missing values into empty strings,
    ensures every value is treated as text, change one or more whitespace characters into normal space
    Ex:
    "body mass\nindex" → "body mass index"
    "child   BMI"      → "child BMI"
    and lastly, remove the unwanted characters from the beginning and end
    
    """
    #and apply test limit, as i already put it in argparse but script does not use it yet 
    if limit is not None:
        variable_annotations = variable_annotations.head(limit).copy()
    return variable_annotations

def build_unique_queries(variable_annotations):
    unique_queries = (
        variable_annotations[["query_text"]].drop_duplicates().reset_index(drop=True)
    ) #double brackets is a dataframe
    return unique_queries

def load_dictionary_index(vector_path, metadata_path):
    if not os.path.exists(vector_path):
        raise FileNotFoundError(f"Dictionary vector file not found: {vector_path}")
    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"Dictionary metadata not found: {metadata_path}")
    vectors = np.load(vector_path, mmap_mode="r")
    metadata = pd.read_csv(metadata_path)
    if vectors.ndim != 2:
        raise ValueError(
            "Dictionary vectors must be a two-dimensional matrix. "
            f"Received shape: {vectors.shape}"
        )
    
    if len(vectors) != len(metadata):
        raise ValueError(
            "Vector and metadata row counts do not match. "
            f"Vectors: {len(vectors):,}; "
            f"metadata: {len(metadata):,}."
        )

    if vectors.shape[1] != EXPECTED_EMBEDDING_DIM:
        raise ValueError(
            "Vector and metadata row counts do not match. "
            f"Vectors: {len(vectors):,}; "
            f"metadata: {len(metadata):,}."
        )
    print(f"Dictionary vector shape: {vectors.shape}")
    print(f"Dictionary metadata shape: {metadata.shape}")

    return vectors, metadata

def main():
    args = parse_arguments()
    
    annotations = load_annotations(args.annotations)
    
    variable_annotations = prepare_variable_annotations(annotations, args.limit)
    
    print(f"Variable rows selected: "f"{len(variable_annotations):,}")
    
    preview = variable_annotations[
        [
            "annotations_id", 
            "Codes", 
            "variable_role", 
            "Content", 
            "query_text"
        ]
    ].head()
    print(preview.to_string(index=False)) 
    
    unique_queries = build_unique_queries(variable_annotations)
    print(f"Unique query texts:" f"{len(unique_queries):,}")
    expected_unique_count = (variable_annotations["query_text"].nunique())
    
    actual_unique_count = len(unique_queries)
    dup_rows_removed = (variable_annotations["query_text"].duplicated().sum())
    duplicates_remaining = (
    unique_queries["query_text"].duplicated().sum()
)
    
    print()
    print(f"Variable annotation rows: {len(variable_annotations):,}")
    print(f"Expected unique queries: {expected_unique_count:,}")
    print(f"Actual unique queries: {actual_unique_count:,}")
    print(f"Repeated rows removed: {dup_rows_removed:,}")
    print(f"Duplicates remaining: {duplicates_remaining:,}")

    vectors, metadata = load_dictionary_index(args.vectors, args.metadata)
    
    for batch_number, (
        start,
        end,
        batch
    ) in enumerate(iter_query_batches(unique_queries, args.batch_size), start=1):
            print(
        f"Batch {batch_number}: "
        f"positions {start} through {end - 1}; "
        f"{len(batch)} queries"
    )

if __name__ == "__main__":
    main()