import argparse
import os
import time
import json
import random
import pandas as pd
from tqdm import tqdm

import anthropic
from anthropic import Anthropic

from utils import (
    SYS_INST, ZERO_SHOT_PROMPT, CoT_PROMPT_TEMPLATE, ONESHOT_ASSISTANT, ONESHOT_USER, 
    TWOSHOT_USER, TWOSHOT_ASSISTANT, SEMANTIC_CODE_PROMPT, CONTROLLED_CHOICE_PROMPT
)

# Securely fetch API key from environment variables
API_KEY = "YOUR_API_KEY_HERE"
if not API_KEY:
    raise ValueError("Missing Anthropic API Key. Set ANTHROPIC_API_KEY in environment variables.")

# Masked API key logging for debugging
print(f"Using API Key: {API_KEY[:4]}...{API_KEY[-4:]} (masked for security)")

client = Anthropic(api_key=API_KEY)

def get_claude_chat(prompt, args, retries=3):
    """
    Calls Claude API with the given prompt and settings.
    Implements a retry mechanism for handling errors.
    """
    messages = []  # No system message inside messages

    if args.prompt_strategy == "std_cls":  # Zero-shot execution
        prompt_text = ZERO_SHOT_PROMPT.replace("{diff_code}", prompt["diff_code"])
        messages.append({"role": "user", "content": prompt_text})

    if args.prompt_strategy == "controled_cls":  # Zero-shot execution
        prompt_text = CONTROLLED_CHOICE_PROMPT.replace("{diff_code}", prompt["diff_code"])
        messages.append({"role": "user", "content": prompt_text})

    elif args.prompt_strategy == "few_shot":  # Few-shot execution
        messages.extend([
            {"role": "user", "content": ONESHOT_USER},
            {"role": "assistant", "content": ONESHOT_ASSISTANT},
            {"role": "user", "content": TWOSHOT_USER},
            {"role": "assistant", "content": TWOSHOT_ASSISTANT},
            {"role": "user", "content": f"Now analyze the following patch:\n```\n{prompt['diff_code']}\n```"}
        ])

    elif args.prompt_strategy == "cot":  # Chain-of-Thought execution
        prompt_text = CoT_PROMPT_TEMPLATE.replace("{diff_code}", prompt["diff_code"])
        messages.append({"role": "user", "content": prompt_text})

    elif args.prompt_strategy == "semantic_code":  # SEMANTIC CODE execution
        diff_code = prompt["diff_code"]
        queries = prompt.get("queries", {})  # Ensure `queries` is a dictionary

        formatted_prompt = SEMANTIC_CODE_PROMPT.replace("{diff_code}", diff_code)
        for key in [
            "added_function_calls", "removed_function_calls", 
            "added_variables", "removed_variables", 
            "added_control_structures", "removed_control_structures"
        ]:
            values = queries.get(key, [])
            formatted_values = ", ".join(values) if values else "None"
            formatted_prompt = formatted_prompt.replace(f"{{{key}}}", formatted_values)

        messages.append({"role": "user", "content": formatted_prompt})

    # API Call with retry mechanism
    for attempt in range(retries):
        try:
            response = client.messages.create(
                model=args.model,
                max_tokens=args.max_gen_length,
                temperature=args.temperature,
                system=SYS_INST,  
                messages=messages
            )
            return "\n".join([block.text for block in response.content])  # Extract response text

        except (anthropic.RateLimitError, anthropic.APIError) as error:
            if attempt < retries - 1:
                sleep_time = min(60, 2 ** attempt + random.uniform(1, 3))  # Exponential backoff
                print(f"API Error: {error}. Retrying in {sleep_time:.2f} seconds...")
                time.sleep(sleep_time)
            else:
                print(f"API Error: {error}. Max retries reached.")
                return None

def run_tests(df, prompt_style, output_file, args):
    """
    Runs tests for a given dataset using the specified prompt style and saves results to JSON in real-time.
    Ensures that responses are stored as they arrive, preventing data loss in case of interruptions.
    """
    if df.empty:
        print(f"Skipping {prompt_style} test: dataset is empty.")
        return

    args.prompt_strategy = prompt_style

    # Ensure the output file exists (or create it)
    if not os.path.exists(output_file):
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump([], f)

    print(f"Running {prompt_style} test on {len(df)} samples...")

    for index, row in tqdm(df.iterrows(), total=len(df)):
        response = get_claude_chat({"diff_code": row["diff_code"]}, args)
        result_entry = {"index": index, "commit_id": row["commit_id"], "response": response}

        # Append result to the JSON file after each response
        with open(output_file, "r+", encoding="utf-8") as f:
            try:
                existing_results = json.load(f)  # Load existing results
            except json.JSONDecodeError:
                existing_results = []  # If file is empty or corrupted, start fresh

            existing_results.append(result_entry)  # Append new response

            f.seek(0)  # Move pointer to start of file
            json.dump(existing_results, f, indent=4)  # Save updated list
            f.truncate()  # Remove remaining old content if overwritten


def main():
    """
    Main function to parse arguments, load datasets, and execute tests.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default="claude-3-haiku-20240307", 
                        choices=["claude-3-haiku-20240307"], 
                        help='Model name')
    parser.add_argument('--data_path', type=str, required=True, help='Path to JSON file for PatchDB')
    parser.add_argument('--semantic_data_path', type=str, required=True, help='Path to JSON file for semantic analysis')
    parser.add_argument('--output_folder', type=str, required=True, help='Output folder for results')
    parser.add_argument('--temperature', type=float, default=0.0, help='Sampling temperature')
    parser.add_argument('--max_gen_length', type=int, default=1024, help='Maximum tokens for generation')
    parser.add_argument('--fewshot_eg', action="store_true", help='Use few-shot examples')
    args = parser.parse_args()

    # Ensure output folder exists
    os.makedirs(args.output_folder, exist_ok=True)

    # Load datasets
    df = pd.read_json(args.data_path, encoding="utf-8")
    semantic_df = pd.read_json(args.semantic_data_path, encoding="utf-8")

    # Run tests with different prompt strategies
    for prompt_style in ["std_cls", "controled_cls"]:
        dataset = semantic_df if prompt_style == "semantic_code" else df  # Select correct dataset
        output_file = os.path.join(args.output_folder, f"{prompt_style}_results.json")
        run_tests(dataset, prompt_style, output_file, args)

    print("All tests completed.")

if __name__ == "__main__":
    main()