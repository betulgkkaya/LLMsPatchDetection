import argparse 
import os
import time
import json
import pandas as pd
from tqdm import tqdm

import openai
from openai import OpenAI
from openai._types import NOT_GIVEN

from utils import (
    SYS_INST, ZERO_SHOT_PROMPT, CoT_PROMPT_TEMPLATE, ONESHOT_ASSISTANT, ONESHOT_USER, 
    TWOSHOT_USER, TWOSHOT_ASSISTANT, SEMANTIC_CODE_PROMPT
)
import tiktoken

client = OpenAI(API_KEY = "YOUR_API_KEY_HERE", base_url="https://api.deepseek.com")


def get_deepseek_chat(prompt, args):
    """
    Calls DeepSeek API with the given prompt and settings.
    """
    messages = []

    

    if args.prompt_strategy == "std_cls":  # Zero-shot execution
        messages.append({"role": "system", "content": SYS_INST})
        prompt_text = ZERO_SHOT_PROMPT.replace("{diff_code}", prompt["diff_code"])
        messages.append({"role": "user", "content": prompt_text})

    elif args.fewshot_eg:  # Few-shot execution
        messages = [
            {"role": "system", "content": SYS_INST},
            {"role": "user", "content": ONESHOT_USER},
            {"role": "assistant", "content": ONESHOT_ASSISTANT},
            {"role": "user", "content": TWOSHOT_USER},
            {"role": "assistant", "content": TWOSHOT_ASSISTANT},
            {"role": "user", "content": prompt["diff_code"]}   
        ]

    elif args.prompt_strategy == "cot":  # Chain-of-Thought execution
        messages.append({"role": "system", "content": SYS_INST})
        prompt_text = CoT_PROMPT_TEMPLATE.replace("{diff_code}", prompt["diff_code"])
        messages.append({"role": "user", "content": prompt_text})

    elif args.prompt_strategy == "semantic_code":  # SEMANTIC CODE
        if "queries" not in prompt:
            raise KeyError("The 'queries' field is missing from the input prompt.")

        queries = prompt["queries"]  # Ensures safe access

        messages.append({"role": "system", "content": SYS_INST})

        diff_code = prompt["diff_code"]
        prompt_text = SEMANTIC_CODE_PROMPT.replace("{diff_code}", diff_code)

        for key in [
            "added_function_calls", "removed_function_calls", "added_variables", 
            "removed_variables", "added_control_structures", "removed_control_structures"
        ]:
            prompt_text = prompt_text.replace(f"{{{key}}}", ", ".join(queries.get(key, [])))

        messages.append({"role": "user", "content": prompt_text})

    try:
        response = client.chat.completions.create(
            model=args.model,
            messages=messages,
            max_tokens=args.max_gen_length,
            temperature=args.temperature,
        )
        
        return response.choices[0].message.content

    except Exception as error:
        print(f"DeepSeek API Error: {error}")
        return None


def run_tests(df, prompt_style, output_file, args):
    """
    Runs tests on the dataset and saves results.
    """
    args.prompt_strategy = prompt_style

    # Load previous results if file exists
    if os.path.exists(output_file) and os.stat(output_file).st_size > 0:
        with open(output_file, "r", encoding="utf-8") as f:
            try:
                data = json.load(f)
                if not isinstance(data, list):
                    raise ValueError("Invalid JSON format")
            except json.JSONDecodeError:
                print("Error: JSON file is corrupted. Creating a new file.")
                data = []
    else:
        data = []

    print(f"Running {prompt_style} test on {len(df)} samples...")

    with open(output_file, "w", encoding="utf-8") as f:
        for index, row in tqdm(df.iterrows(), total=len(df)):
            # Ensure "queries" exists before proceeding


            response = get_deepseek_chat({
                "diff_code": row["diff_code"]
            }, args)

            result = {"index": index, "commit_id": row["commit_id"], "response": response}
            data.append(result)

            # Write results to file
            f.seek(0)
            json.dump(data, f, indent=4, ensure_ascii=False)
            f.flush()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default="deepseek-chat", choices=["deepseek-chat", "deepseek-reasoner"], help='Model name')
    parser.add_argument('--data_path', type=str, required=True, help='Path to JSON file for PatchDB')
    parser.add_argument('--semantic_data_path', type=str, required=True, help='Path to JSON file for semantic analysis')
    parser.add_argument('--output_folder', type=str, required=True, help='Output folder')
    parser.add_argument('--temperature', type=float, default=0.0, help='Sampling temperature')
    parser.add_argument('--max_gen_length', type=int, default=512)
    parser.add_argument('--seed', type=int, default=12345)
    parser.add_argument('--logprobs', action="store_true", help='Return logprobs')
    parser.add_argument('--fewshot_eg', action="store_true", help='Use few-shot examples')
    args = parser.parse_args()
    
    df = pd.read_json(args.data_path, encoding="utf-8")
    semantic_df = pd.read_json(args.semantic_data_path, encoding="utf-8")

    for prompt_style in ["std_cls","few_shot","cot", "semantic_code"]:
        dataset = semantic_df if prompt_style == "semantic_code" else df
        output_file = os.path.join(args.output_folder, f"{prompt_style}_results.json")
        run_tests(dataset, prompt_style, output_file, args)

    print("All tests completed.")


if __name__ == "__main__":
    main()
