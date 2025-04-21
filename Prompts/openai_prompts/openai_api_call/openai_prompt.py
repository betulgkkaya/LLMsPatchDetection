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

# Add your OpenAI API key here
client = OpenAI(api_key="YOUR_API_KEY")


def truncate_tokens_from_messages(messages, model, max_gen_length):
    """
    Truncate messages if the token count exceeds the model's limit.
    """
    model_token_limits = {
        "gpt-3.5-turbo": 16385,
    }
    max_tokens = model_token_limits.get(model, 4096) - max_gen_length

    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        print("Warning: Model not found. Using cl100k_base encoding.")
        encoding = tiktoken.get_encoding("cl100k_base")

    tokens_per_message = 3
    num_tokens = 3  
    trunc_messages = []

    for message in messages:
        tm = {}
        num_tokens += tokens_per_message

        for key, value in message.items():
            encoded_value = encoding.encode(value)
            num_tokens += len(encoded_value)

            if num_tokens > max_tokens:
                print(f"Truncating message: {value[:100]}...")
                tm[key] = encoding.decode(encoded_value[:max_tokens - num_tokens])
                break
            else:
                tm[key] = value
        trunc_messages.append(tm)
    
    return trunc_messages


def get_openai_chat(prompt, args):
    """
    Calls OpenAI API with the given prompt and settings.
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
        messages.append({"role": "system", "content": SYS_INST})

        # Extract structured metadata from prompt
        diff_code = prompt["diff_code"]
        queries = prompt["queries"]

        # Fill in SEMANTIC_CODE_PROMPT
        prompt_text = SEMANTIC_CODE_PROMPT.replace("{diff_code}", diff_code)
        for key in ["added_function_calls", "removed_function_calls", "added_variables", "removed_variables", "added_control_structures", "removed_control_structures"]:
            prompt_text = prompt_text.replace(f"{{{key}}}", ", ".join(queries.get(key, [])))

        messages.append({"role": "user", "content": prompt_text})

    messages = truncate_tokens_from_messages(messages, args.model, args.max_gen_length)

    try:
        response = client.chat.completions.create(
            model=args.model,
            messages=messages,
            max_tokens=args.max_gen_length,
            temperature=args.temperature,
            seed=args.seed,
            logprobs=args.logprobs,
            top_logprobs=5 if args.logprobs else NOT_GIVEN,
        )
        
        return response.choices[0].message.content

    except (openai.RateLimitError, openai.APITimeoutError, openai.APIConnectionError) as error:
        retry_time = error.retry_after if hasattr(error, "retry_after") else 5
        print(f"Rate Limit or Connection Error. Sleeping for {retry_time} seconds ...")
        time.sleep(retry_time)
        return get_openai_chat(prompt, args)

    except openai.BadRequestError as error:
        print(f"Bad Request Error: {error}")
        return None


def run_tests(df, prompt_style, output_file, args):
    args.prompt_strategy = prompt_style
    
    # Check if the file exists and load existing JSON
    if os.path.exists(output_file) and os.stat(output_file).st_size > 0:
        with open(output_file, "r", encoding="utf-8") as f:
            try:
                data = json.load(f)  # Load existing JSON array
                if not isinstance(data, list):  # Ensure it's a list
                    raise ValueError("Invalid JSON format")
            except json.JSONDecodeError:
                print("Error: JSON file is corrupted or not in array format. Creating a new file.")
                data = []
    else:
        data = []  # Initialize empty list if file doesn't exist

    print(f"Running {prompt_style} test on {len(df)} samples...")

    with open(output_file, "w", encoding="utf-8") as f:  # Open in write mode to update JSON array
        for index, row in tqdm(df.iterrows(), total=len(df)):
            response = None  # Initialize response
            
            if prompt_style == "std_cls":  # Zero-Shot
                prompt_text = ZERO_SHOT_PROMPT.replace("{diff_code}", row["diff_code"])
                response = get_openai_chat({"diff_code": row["diff_code"]}, args)

            elif prompt_style == "cot":  # Chain-of-Thought
                prompt_text = CoT_PROMPT_TEMPLATE.replace("{diff_code}", row["diff_code"])
                response = get_openai_chat({"diff_code": row["diff_code"]}, args)

            elif prompt_style == "few_shot":  # Few-Shot
                args.fewshot_eg = True
                prompt_text = row["diff_code"]
                response = get_openai_chat({"diff_code": row["diff_code"]}, args)

            elif prompt_style == "semantic_code":  # Semantic Analysis
                queries = row.get("queries", {})
                prompt_text = SEMANTIC_CODE_PROMPT.replace("{diff_code}", row["diff_code"])

                for key in ["added_function_calls", "removed_function_calls", "added_variables", 
                            "removed_variables", "added_control_structures", "removed_control_structures"]:
                    prompt_text = prompt_text.replace(f"{{{key}}}", ", ".join(queries.get(key, [])))


                response = get_openai_chat({"diff_code": row["diff_code"], "queries": queries}, args)

            # Create result dictionary
            result = {"index": index, "commit_id": row["commit_id"], "response": response}
            data.append(result)  # Append new entry to list

            # Write updated JSON array to file
            f.seek(0)  # Move to the beginning of the file
            json.dump(data, f, indent=4, ensure_ascii=False)
            f.flush()  # Ensure immediate write to disk

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default="gpt-3.5-turbo", choices=["gpt-3.5-turbo"], help='Model name')
    parser.add_argument('--data_path', type=str, required=True, help='Path to JSON file for PatchDB')
    parser.add_argument('--semantic_data_path', type=str, required=True, help='Path to JSON file for semantic analysis')
    parser.add_argument('--output_folder', type=str, required=True, help='Output folder')
    parser.add_argument('--temperature', type=float, default=0.0, help='Sampling temperature')
    parser.add_argument('--max_gen_length', type=int, default=1024)
    parser.add_argument('--seed', type=int, default=12345)
    parser.add_argument('--logprobs', action="store_true", help='Return logprobs')
    parser.add_argument('--fewshot_eg', action="store_true", help='Use few-shot examples')
    args = parser.parse_args()
    
    # Load datasets
    df = pd.read_json(args.data_path, encoding="utf-8")
    semantic_df = pd.read_json(args.semantic_data_path, encoding="utf-8")

    # Run tests for all prompt styles, ensuring semantic_df is used only for semantic_code
    for prompt_style in ["std_cls","few_shot","cot", "semantic_code"]:
        dataset = semantic_df if prompt_style == "semantic_code" else df  # Select correct dataset
        output_file = os.path.join(args.output_folder, f"{prompt_style}_results.json")
        run_tests(dataset, prompt_style, output_file, args)

    print("All tests completed.")


if __name__ == "__main__":
    main()


