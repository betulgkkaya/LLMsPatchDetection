import os
import json
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# Ensure the script is looking for the correct dataset
dataset_path = "patch_db_balanced_sample.json"

if not os.path.exists(dataset_path):
    raise FileNotFoundError(f"Dataset file '{dataset_path}' not found in output_results/")

# Load actual labels
with open(dataset_path, "r", encoding="utf-8") as f:
    actual_data = json.load(f)

# Define all expected result files
result_files = [
    "std_cls_results_4o.json",
    "semantic_code_results_4o.json",
    "cot_results_4o.json",
    "few_shot_results_4o.json"
    
]
# Check which files exist
existing_files = [f for f in result_files if os.path.exists(f)]

if not existing_files:
    raise FileNotFoundError("No result files found in output_results/. Ensure the script has generated them.")

# Convert actual data to DataFrame
actual_df = pd.DataFrame(actual_data)

# Ensure index is an integer and create a unique key (commit_id + index)
actual_df["index"] = actual_df["index"].astype(int)
actual_df["unique_id"] = actual_df["commit_id"] + "_" + actual_df["index"].astype(str)

# Debug: Check dataset sizes before processing
print(f"\nLoaded Actual Dataset: {len(actual_df)} records")
print("Unique commit_id + index combinations in actual dataset:", actual_df["unique_id"].nunique())

# Iterate through all existing result files
evaluation_results_all = {}

for file_name in existing_files:
    prompt_style = file_name.replace("_results.json", "")
    with open(file_name, "r", encoding="utf-8") as f:
        predicted_data = json.load(f)  # Load entire JSON object as a list

    # Convert predicted data to DataFrame
    predicted_df = pd.DataFrame(predicted_data)

    # Ensure index is an integer and create a unique key (commit_id + index)
    predicted_df["index"] = predicted_df["index"].astype(int)
    predicted_df["unique_id"] = predicted_df["commit_id"] + "_" + predicted_df["index"].astype(str)

    # Debug: Check dataset sizes before merging
    print(f"\nProcessing {file_name} ...")
    print(f"Predicted dataset size: {len(predicted_df)}")
    print("Unique commit_id + index combinations in predicted dataset:", predicted_df["unique_id"].nunique())

    # Check for duplicate unique_id in actual_df and predicted_df
    actual_duplicates = actual_df[actual_df.duplicated(subset=["unique_id"], keep=False)]
    predicted_duplicates = predicted_df[predicted_df.duplicated(subset=["unique_id"], keep=False)]

    print("\n🔍 Checking for duplicate unique_id (commit_id + index)...")
    print(f"Duplicate unique_id in actual dataset: {actual_duplicates['unique_id'].nunique()} unique duplicates")
    if not actual_duplicates.empty:
        print("Sample duplicate unique_id in actual dataset:")
        print(actual_duplicates["unique_id"].value_counts().head(10))  # Show up to 10 duplicate IDs

    print(f"Duplicate unique_id in predicted dataset: {predicted_duplicates['unique_id'].nunique()} unique duplicates")
    if not predicted_duplicates.empty:
        print("Sample duplicate unique_id in predicted dataset:")
        print(predicted_duplicates["unique_id"].value_counts().head(10))  # Show up to 10 duplicate IDs

    # Ensure `unique_id` is unique before merging
    actual_df = actual_df.drop_duplicates(subset=["unique_id"])
    predicted_df = predicted_df.drop_duplicates(subset=["unique_id"])

    # Rename actual category column
    actual_df = actual_df.rename(columns={"category": "category_actual"})
    predicted_df = predicted_df.rename(columns={"response": "response_predicted"})

    # Merge data using unique_id
    merged_df = actual_df.merge(predicted_df, on="unique_id", how="inner")

    # Debug: Check merged dataset size
    print(f"Merged dataset size: {len(merged_df)} (should be close to the actual dataset size)")
    print("Merged dataset preview:")
    print(merged_df.head())

    # Function to standardize responses
    def extract_label(response):
        if not isinstance(response, str):
            return -1  # Invalid response

        response = response.strip().upper()

        # Check if response contains keywords
        if "1" in response or "YES" in response:
            return 1
        elif "2" in response or "NO" in response:
            return 0

        return -1  # If the format is unexpected

    # Map actual labels and extract predicted labels
    y_true = merged_df["category_actual"].map({"security": 1, "non-security": 0})
    y_pred = merged_df["response_predicted"].apply(extract_label)

    # Debug: Check dataset size before filtering invalid predictions
    print(f"Total samples before filtering: {len(y_pred)}")
    print(f"Invalid (-1) predictions count: {sum(y_pred == -1)}")

    # Filter invalid responses
    valid_indices = y_pred != -1
    y_true = y_true[valid_indices]
    y_pred = y_pred[valid_indices]

    # Debug: Check dataset size after filtering
    print(f"Valid samples after filtering: {len(y_true)} (should be close to actual dataset size)")

    # Compute metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    conf_matrix = confusion_matrix(y_true, y_pred)

    # Debug: Check sum of confusion matrix
    print(f"Sum of confusion matrix: {conf_matrix.sum()} (should match valid sample count)")

    # Store results
    evaluation_results_all[prompt_style] = {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "confusion_matrix": conf_matrix.tolist()
    }

# Save evaluation results
with open("evaluation_results_all.json", "w", encoding="utf-8") as f:
    json.dump(evaluation_results_all, f, indent=4)

print("\n Evaluation results saved to evaluation_results_all.json")
