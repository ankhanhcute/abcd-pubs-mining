"""
embed_data_dictionary.py

Reads the ABCD data dictionary, constructs meaningful text for each variable,
creates embeddings in batches, and saves vectors with corresponding metadata.

Outputs:
    <output>.npy
    <output>_meta.csv
    <output>_progress.json  (exists only while an incomplete run is in progress)

Example:
    python embed_data_dictionary.py \
        --input data_dictionary_levels.xlsx \
        --sheet "data dictionary" \
        --output dict_embeddings

Test with only 10 rows:
    python embed_data_dictionary.py \
        --input data_dictionary_levels.xlsx \
        --output test_embeddings \
        --limit 10

Requirements:
    pip install openai pandas numpy openpyxl

Environment variable:
    export OPENAI_API_KEY="your-api-key"
"""

import argparse
import json
import os
import sys
import time
import hashlib

import numpy as np
import pandas as pd
from openai import OpenAI


MODEL = "text-embedding-3-small"
EMBEDDING_DIM = 1536
BATCH_SIZE = 500
SLEEP_BETWEEN_CALLS = 0.5
MAX_RETRIES = 4


# These names explain what each value means inside the embedding text.
DISPLAY_NAMES = {
    "label": "Variable",
    "table_label": "Assessment",
    "domain": "Domain",
    "sub_domain": "Subdomain",
    "source": "Source",
    "metric": "Metric",
    "atlas": "Atlas",
    "unit": "Unit",
    "type_var": "Variable type",
}


# These columns are embedded by default.
DEFAULT_TEXT_COLUMNS = [
    "label",
    "table_label",
    "domain",
    "sub_domain",
    "source",
    "metric",
    "atlas",
    "unit",
    "type_var",
]

def calculate_dataset_hash(df):
    hasher = hashlib.sha256()
    for text in df["_embed_text"]:
        hasher.update(text.encode("utf-8"))
        hasher.update(b"\0")
    return hasher.hexdigest()

def create_run_config(df, args):
    return {
        "model": MODEL,
        "total_rows": len(df),
        "embedding_dimension": EMBEDDING_DIM,
        "text_columns": list(args.text_columns),
        "exclude_administrative": args.exclude_administrative,
        "dataset_hash": calculate_dataset_hash(df),
    }

def build_embed_text(row, text_columns):
    """
    Build one labeled text description from one data-dictionary row.

    Missing and empty values are skipped.
    """
    parts = []

    for column_name in text_columns:
        value = row.get(column_name)

        if pd.notna(value) and str(value).strip():
            clean_value = str(value).strip()

            display_name = DISPLAY_NAMES.get(
                column_name,
                column_name
            )

            text_part = f"{display_name}: {clean_value}"
            parts.append(text_part)

    return ". ".join(parts)


def initialize_or_resume(output_prefix, df, run_config):
    vector_path = f"{output_prefix}.npy"
    metadata_path = f"{output_prefix}_meta.csv"
    progress_path = f"{output_prefix}_progress.json"
    
    vector_exists = os.path.exists(vector_path)
    metadata_exists = os.path.exists(metadata_path)
    progress_exists = os.path.exists(progress_path)
    # No output exists: initialize a new run
    if not vector_exists and not metadata_exists and not progress_exists:
        print("No previous output found. Initializing a new run.")
        
        save_metadata_once(df, metadata_path)
        
        vectors = np.lib.format.open_memmap(
            vector_path, 
            mode="w+", 
            dtype=np.float32,
            shape=(
                run_config["total_rows"],
                run_config["embedding_dimension"],
            )
        )
        vectors.flush()
        save_progress(progress_path, run_config, rows_done=0)
        return vectors, 0 
    if vector_exists and metadata_exists and not progress_exists:
        sys.exit("This output prefix already contains a completed run:\n"
            f"  {vector_path}\n"
            f"  {metadata_path}\n"
            "Use a different --output prefix to create another run.")
    if not (
        vector_exists
        and metadata_exists
        and progress_exists
    ):
        sys.exit(
            "ERROR: Incomplete output files detected. "
            "Do not resume because the files may be misaligned."
        )
    # Resume an incomplete run.
    with open(progress_path, "r", encoding="utf-8") as file:
        progress = json.load(file)

    for key, expected_value in run_config.items():
        saved_value = progress.get(key)

        if saved_value != expected_value:
            sys.exit(
                f"ERROR: Checkpoint setting {key!r} changed.\n"
                f"Saved: {saved_value}\n"
                f"Current: {expected_value}"
            )

    rows_done = progress.get("rows_done")

    if not isinstance(rows_done, int):
        sys.exit(
            "ERROR: Checkpoint rows_done is missing or invalid."
        )

    if not 0 <= rows_done <= run_config["total_rows"]:
        sys.exit(
            "ERROR: Checkpoint rows_done is outside the valid range."
        )

    vectors = np.load(
        vector_path,
        mmap_mode="r+"
    )

    expected_shape = (
        run_config["total_rows"],
        run_config["embedding_dimension"],
    )

    if vectors.shape != expected_shape:
        sys.exit(
            f"ERROR: Vector shape is {vectors.shape}, "
            f"but expected {expected_shape}."
        )

    metadata_count = len(
        pd.read_csv(metadata_path, usecols=["name"])
    )

    if metadata_count != run_config["total_rows"]:
        sys.exit(
            "ERROR: Metadata row count does not match "
            "the prepared dataset."
        )

    print(
        f"Resuming from checkpoint: "
        f"{rows_done:,} rows completed."
    )

    return vectors, rows_done
    
        
        

def save_progress(
    progress_path, 
    run_config,
    rows_done
):
    """
    Save the current vectors, metadata, and completed row count.
    """
    
    progress = {
        **run_config, 
        "rows_done": rows_done,
    }
    temp_path = f"{progress_path}.tmp"
    with open(temp_path, "w", encoding="utf-8") as file:
        json.dump(progress, file, indent=2)
    os.replace(temp_path, progress_path)
def save_metadata_once(df, metadata_path):
    metadata_columns = [
        "name",
        "label",
        "domain",
        "sub_domain",
        "source",
        "table_name",
        "table_label",
        "metric",
        "atlas",
        "unit",
        "type_var",
        "type_data",
        "type_level",
        "_embed_text",
    ]
    existing_columns = [
        column
        for column in metadata_columns
        if column in df.columns
    ]
    metadata = df[existing_columns].copy()
    metadata = metadata.rename(
        columns={"_embed_text": "embed_text"}
    )
    metadata.to_csv(
        metadata_path, 
        index=False
    )
def embed_batch(client, texts):
    """
    Send one batch of text to the OpenAI embeddings API.

    The request is retried when an error occurs.

    Returns:
        A list of embedding vectors if successful.
        None if all attempts fail.
    """
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = client.embeddings.create(
                model=MODEL,
                input=texts
            )

            vectors = [
                item.embedding
                for item in response.data
            ]

            return vectors

        except Exception as error:
            print(
                f"[warn] Embedding call failed "
                f"(attempt {attempt}/{MAX_RETRIES}): "
                f"{error}"
            )

            wait_time = SLEEP_BETWEEN_CALLS * attempt
            time.sleep(wait_time)

    print(
        "[error] Batch failed after all retries. "
        "Stopping without losing completed batches."
    )

    return None


def main():
    parser = argparse.ArgumentParser(
        description="Embed the ABCD data dictionary for lookup-based matching."
    )
    parser.add_argument("--input", required=True, help="Path to data_dictionary_levels.xlsx")
    parser.add_argument("--sheet", default="data dictionary", help="Excel sheet to read")
    parser.add_argument("--text-columns", nargs="+", default=DEFAULT_TEXT_COLUMNS)
    parser.add_argument("--output", default="dict_embeddings", help="Output file prefix")
    parser.add_argument("--limit", type=int, default=None, help="Random test-sample size")
    parser.add_argument(
        "--exclude-administrative",
        action="store_true",
        help="Exclude administrative variables"
    )
    args = parser.parse_args()

    # Check API key.
    if not os.environ.get("OPENAI_API_KEY"):
        sys.exit("ERROR: OPENAI_API_KEY is not set.")

    # Read the Excel dictionary.
    print(f"Reading sheet {args.sheet!r} from {args.input}...")

    try:
        df = pd.read_excel(args.input, sheet_name=args.sheet)
    except FileNotFoundError:
        sys.exit(f"ERROR: Input file not found: {args.input}")
    except ValueError as error:
        sys.exit(f"ERROR: Could not read sheet {args.sheet!r}: {error}")

    print(f"Loaded {len(df):,} rows and {len(df.columns)} columns.")

    # Check that required columns exist.
    required_columns = set(args.text_columns)
    required_columns.add("name")

    missing_columns = [
        column for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        sys.exit(
            f"ERROR: Missing columns: {missing_columns}\n"
            f"Available columns: {list(df.columns)}"
        )

    # Optionally remove administrative variables.
    if args.exclude_administrative:
        if "type_var" not in df.columns:
            sys.exit("ERROR: type_var is required to exclude administrative variables.")

        original_count = len(df)
        clean_type = df["type_var"].fillna("").astype(str).str.strip().str.lower()
        df = df[clean_type != "administrative"].copy()

        print(f"Excluded {original_count - len(df):,} administrative variables.")

    # Build the text that will be embedded.
    print("Building embedding text...")

    df["_embed_text"] = df.apply(
        lambda row: build_embed_text(row, args.text_columns),
        axis=1
    )

    # Remove rows that produced empty text.
    valid_mask = df["_embed_text"].fillna("").str.strip() != ""
    number_skipped = int((~valid_mask).sum())

    if number_skipped:
        print(f"Skipped {number_skipped:,} rows with empty embedding text.")

    df = df[valid_mask].reset_index(drop=True)

    if df.empty:
        sys.exit("ERROR: No valid rows remain to embed.")

    # Use a reproducible random sample when testing.
    if args.limit is not None:
        if args.limit <= 0:
            sys.exit("ERROR: --limit must be greater than zero.")

        test_size = min(args.limit, len(df))
        df = df.sample(n=test_size, random_state=42).reset_index(drop=True)

        print(f"Testing with a random sample of {len(df):,} rows.")

    # Show a few examples before calling the API.
    print("\nEmbedding text examples:\n")

    for index in range(min(3, len(df))):
        print(f"Example {index + 1}:")
        print(df.iloc[index]["_embed_text"])
        print()

    # Create the run configuration.
    run_config = create_run_config(df, args)
    total_rows = len(df)

    # Create paths and output directory.
    progress_path = f"{args.output}_progress.json"
    output_directory = os.path.dirname(os.path.abspath(args.output))
    os.makedirs(output_directory, exist_ok=True)

    # Initialize a new memory-mapped file or resume.
    vectors, rows_done = initialize_or_resume(
        args.output,
        df,
        run_config
    )

    if rows_done > total_rows:
        sys.exit(
            f"ERROR: Checkpoint has {rows_done:,} completed rows, "
            f"but the dataset has only {total_rows:,} rows."
        )

    # Calculate the remaining work.
    rows_remaining = total_rows - rows_done
    number_of_batches = (rows_remaining + BATCH_SIZE - 1) // BATCH_SIZE

    print(
        f"{rows_remaining:,} rows remain in {number_of_batches:,} "
        f"batches of up to {BATCH_SIZE}."
    )

    client = OpenAI()

    # Embed each unfinished batch.
    for batch_number, start in enumerate(
        range(rows_done, total_rows, BATCH_SIZE),
        start=1
    ):
        end = min(start + BATCH_SIZE, total_rows)
        batch_texts = df.iloc[start:end]["_embed_text"].tolist()

        print(
            f"Batch {batch_number}/{number_of_batches}: "
            f"rows {start:,} through {end - 1:,}"
        )

        batch_vectors = embed_batch(client, batch_texts)

        if batch_vectors is None:
            print(f"Stopped safely after {rows_done:,} completed rows.")
            print("Run the same command again to resume.")
            sys.exit(1)

        batch_array = np.asarray(batch_vectors, dtype=np.float32)
        expected_shape = (len(batch_texts), EMBEDDING_DIM)

        if batch_array.shape != expected_shape:
            sys.exit(
                f"ERROR: Received shape {batch_array.shape}, "
                f"but expected {expected_shape}."
            )

        # Write the new batch directly to its final positions.
        vectors[start:end] = batch_array
        vectors.flush()

        # Save progress only after the vectors reach the disk.
        rows_done = end
        save_progress(progress_path, run_config, rows_done)

        time.sleep(SLEEP_BETWEEN_CALLS)

    # Verify that everything completed correctly.
    if rows_done != total_rows:
        sys.exit(
            f"ERROR: Completed {rows_done:,} rows, "
            f"but expected {total_rows:,}."
        )

    vectors.flush()

    expected_final_shape = (total_rows, EMBEDDING_DIM)

    if vectors.shape != expected_final_shape:
        sys.exit(
            f"ERROR: Final shape is {vectors.shape}, "
            f"but expected {expected_final_shape}."
        )

    metadata_path = f"{args.output}_meta.csv"
    metadata_count = len(pd.read_csv(metadata_path, usecols=["name"]))

    if metadata_count != total_rows:
        sys.exit(
            f"ERROR: Metadata contains {metadata_count:,} rows, "
            f"but expected {total_rows:,}."
        )

    # Removing the progress file marks the run as complete.
    if os.path.exists(progress_path):
        os.remove(progress_path)

    print(f"\nDone. Embedded {rows_done:,} rows.")
    print(f"Vectors: {args.output}.npy")
    print(f"Metadata: {metadata_path}")
    print(f"Final vector shape: {vectors.shape}")


if __name__ == "__main__":
    main()