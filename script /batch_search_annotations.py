import argparse 
import pandas as pd 

def main():
    parser = argparse.ArgumentParser(
        description="Search the ABCD dictionary for many annotations labels."
    )
    parser.add_argument("--annotations", required=True, help="Path to be the annotation CSV file.")
    parser.add_argument("--limit", type=int, default=None, help="Optional number of unique labels to test.")
    args = parser.parse_args()
    print(f"Reading annotations from: {args.annotations}")
    
    annotations = pd.read_csv(args.annotations)
    annotations.insert(0, "annotations_id", range(len(annotations)))
    print(f"Rows: {len(annotations):,}")
    print(f"Columns: {list(annotations.columns)}")
    code_counts = annotations["Codes"].value_counts(dropna=False)
    print(code_counts.to_string())
    print()
if __name__ == "__main__":
    main()