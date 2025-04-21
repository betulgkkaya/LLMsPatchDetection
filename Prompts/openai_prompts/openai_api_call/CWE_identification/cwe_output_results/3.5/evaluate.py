import os
import json
import pandas as pd
import re
from collections import Counter

# Ensure the script is looking for the correct dataset
dataset_path = "CWE_sample.json"

if not os.path.exists(dataset_path):
    raise FileNotFoundError(f"Dataset file '{dataset_path}' not found.")

# Load actual labels
with open(dataset_path, "r", encoding="utf-8") as f:
    actual_data = json.load(f)

# Define all expected result files
result_files = [
    "std_cls_results_3.5.json",
    "controled_cls_results_3.5.json"
]

# Check which files exist
existing_files = [f for f in result_files if os.path.exists(f)]

if not existing_files:
    raise FileNotFoundError("No result files found. Ensure the script has generated them.")

# Convert actual data to DataFrame
actual_df = pd.DataFrame(actual_data)

# Ensure `commit_id` is a string and `index` is an integer
actual_df["commit_id"] = actual_df["commit_id"].astype(str).str.strip().str.lower()
actual_df["index"] = actual_df["index"].astype(int)
actual_df["unique_id"] = actual_df["commit_id"] + "_" + actual_df["index"].astype(str)

# Extract CWE_ID as a string number
actual_df["CWE_ID"] = actual_df["CWE_ID"].astype(str).str.extract(r"(\d+)")[0]

# Iterate through all existing result files
evaluation_results_all = {}

for file_name in existing_files:
    prompt_style = file_name.replace("_results.json", "")
    
    with open(file_name, "r", encoding="utf-8") as f:
        predicted_data = json.load(f)

    # Convert predicted data to DataFrame
    predicted_df = pd.DataFrame(predicted_data)

    # Ensure `commit_id` is a string and `index` is an integer
    predicted_df["commit_id"] = predicted_df["commit_id"].astype(str).str.strip().str.lower()
    predicted_df["index"] = predicted_df["index"].astype(int)
    predicted_df["unique_id"] = predicted_df["commit_id"] + "_" + predicted_df["index"].astype(str)

    # Function to extract CWE ID from response
    def extract_cwe(response):
        if not isinstance(response, str):
            return None  # Return None for invalid responses
        match = re.search(r"(\d+)", response)  # Extract first numeric CWE ID
        return match.group(1) if match else None

    # Extract CWE ID from predictions
    predicted_df["CWE_predicted"] = predicted_df["response"].apply(extract_cwe)

    # Ensure `unique_id` is unique before merging
    actual_df = actual_df.drop_duplicates(subset=["unique_id"])
    predicted_df = predicted_df.drop_duplicates(subset=["unique_id"])

    # Merge data using unique_id
    merged_df = actual_df.merge(predicted_df, on="unique_id", how="inner")

    # Compute evaluation metrics
    total_cases = len(merged_df)
    correct_cases_df = merged_df[merged_df["CWE_ID"] == merged_df["CWE_predicted"]]
    wrong_cases_df = merged_df[merged_df["CWE_ID"] != merged_df["CWE_predicted"]]
    missed_cases_df = merged_df[merged_df["CWE_predicted"].isna()]

    correct_cases = len(correct_cases_df)
    wrong_cases = len(wrong_cases_df)
    missed_cases = len(missed_cases_df)

    # Count correct CWE identifications
    correct_cwe_counts = correct_cases_df["CWE_ID"].value_counts()
    correct_cwe_percentages = {
        cwe: f"{count} ({(count / correct_cases * 100):.2f}%)"
        for cwe, count in correct_cwe_counts.items()
    } if correct_cases > 0 else {}

    # Count wrong CWE categorization
    wrong_cwe_counts = wrong_cases_df["CWE_predicted"].value_counts()
    wrong_cwe_percentages = {
        cwe: f"{count} ({(count / wrong_cases * 100):.2f}%)"
        for cwe, count in wrong_cwe_counts.items()
    } if wrong_cases > 0 else {}

    # Count missed CWE cases
    missed_cwe_counts = missed_cases_df["CWE_ID"].value_counts()
    missed_cwe_percentages = {
        cwe: f"{count} ({(count / missed_cases * 100):.2f}%)"
        for cwe, count in missed_cwe_counts.items()
    } if missed_cases > 0 else {}

    # Compute total percentages
    correct_percentage = f"{correct_cases} ({(correct_cases / total_cases * 100):.2f}%)" if total_cases > 0 else "0 (0.00%)"
    wrong_percentage = f"{wrong_cases} ({(wrong_cases / total_cases * 100):.2f}%)" if total_cases > 0 else "0 (0.00%)"
    missed_percentage = f"{missed_cases} ({(missed_cases / total_cases * 100):.2f}%)" if total_cases > 0 else "0 (0.00%)"

    # Store results
    evaluation_results_all[prompt_style] = {
        "total_cases": total_cases,
        "total_correct": correct_percentage,
        "correct_cwe_identifications": correct_cwe_percentages,
        "total_wrong": wrong_percentage,
        "wrong_cwe_categorization": wrong_cwe_percentages,
        "total_missed": missed_percentage,
        "missed_cwe_cases": missed_cwe_percentages
    }

# Save evaluation results
with open("evaluation_results_all.json", "w", encoding="utf-8") as f:
    json.dump(evaluation_results_all, f, indent=4)

print("\n✅ Evaluation results saved to evaluation_results_all.json")
