# Ben Kabongo
# LLaMA 3 - For Statement Extraction and Aspect-Based Sentiment Analysis

# April 2025

import argparse
import datasets
import json
import os
import pandas as pd
import re
import time
import torch

from prompt import examples, system_prompt
from transformers import pipeline


parser = argparse.ArgumentParser(description="LLaMA 3 for Statement Extraction and Aspect-Based Sentiment Analysis")
parser.add_argument("--model", type=str, default="llama3/Meta-Llama-3-8B-Instruct", help="Model path")
parser.add_argument("--domain", type=str, default="restaurant", help="Domain for sentiment analysis")
parser.add_argument("--dataset_path", type=str, default="dataset/restaurant.csv", help="Dataset for sentiment analysis")
parser.add_argument("--json_format", action=argparse.BooleanOptionalAction, default=True, help="Dataset format")
parser.add_argument("--output_dir", type=str, default="dataset/", help="Output path for results")
parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
parser.add_argument("--max_new_tokens", type=int, default=512, help="Maximum new tokens")
parser.add_argument("--do_sample", action=argparse.BooleanOptionalAction, default=True, help="Enable sampling")
parser.add_argument("--temperature", type=float, default=0.6, help="Temperature for sampling")
parser.add_argument("--top_p", type=float, default=0.9, help="Top-p sampling probability")
parser.add_argument("--skip_existing", action=argparse.BooleanOptionalAction, default=False, help="Skip existing results")
parser.set_defaults(json_format=True)
parser.set_defaults(do_sample=True)
parser.set_defaults(skip_existing=False)
config = parser.parse_args()


def format_message(example):
    example["messages"] = [
        {"role": "system", "content": system_prompt + examples[config.domain]},
        {"role": "user", "content": "Now analyze the following review: " + example["review"]}
    ]
    return example


def extract_json_from_output(text):
    match = re.search(r"\[\s*{.*?}\s*\]", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            return None
    return None


def main():
    if config.domain not in examples:
        config.domain = "restaurant"

    os.makedirs(config.output_dir, exist_ok=True)
    absa_path = os.path.join(config.output_dir, "absa_filtered.csv")
    
    pipe = pipeline(
        "text-generation",
        model=config.model,
        model_kwargs={"torch_dtype": torch.bfloat16},
        device_map="auto",
        batch_size=config.batch_size,
    )
    pipe.tokenizer.pad_token_id = pipe.tokenizer.eos_token_id
    pipe.tokenizer.padding_side = "left"

    terminators = [
        pipe.tokenizer.eos_token_id,
        pipe.tokenizer.convert_tokens_to_ids("<|eot_id|>")
    ]

    fmt = "json" if config.json_format else "csv"
    dataset = datasets.load_dataset(fmt, data_files={"train": config.dataset_path}, split="train", streaming=True)
    if os.path.exists(absa_path) and config.skip_existing:
        print(f"[INFO] Skipping existing results in {absa_path}")
        print("Before skipping: ", next(iter(dataset)))
        df = pd.read_csv(absa_path)
        length = len(df)
        dataset = dataset.skip(length)
        print("After skipping: ", next(iter(dataset)))
    else:
        print(f"[INFO] No existing results found in {absa_path}")
        
    if config.json_format: # JSON (Amazon Reviews)
        dataset = dataset.rename_colums({"parent_asin": "item_id", "text": "review"})
    dataset = dataset.map(format_message)

    batch_dataset = dataset.batch(batch_size=config.batch_size)
    if os.path.exists(os.path.join(config.output_dir, "state_dict.json")):
        with open(os.path.join(config.output_dir, "state_dict.json"), "r") as f:
            state_dict = json.load(f)
        batch_dataset.load_state_dict(state_dict)
        print(f"[INFO] Loading state dict from {os.path.join(config.output_dir, 'state_dict.json')}")
    else:
        print(f"[INFO] No state dict found!")

    for batch in batch_dataset:
        start = time.time()
        outputs = pipe(
            batch["messages"],
            max_new_tokens=config.max_new_tokens,
            eos_token_id=terminators,
            do_sample=config.do_sample,
            temperature=config.temperature,
            top_p=config.top_p,
        )
        end = time.time()
        print(f"Batch processed in {end - start:.2f} seconds")
        raw_outputs = [out[0]["generated_text"][-1]["content"] for out in outputs]
        json_outputs = [extract_json_from_output(raw) for raw in raw_outputs]
         
        batch["absa"] = json_outputs
        del batch["messages"]
        df = pd.DataFrame(batch)
        df.to_csv(absa_path, mode='a', header=not os.path.exists(absa_path), index=False, escapechar='\\')

        state_dict = batch_dataset.state_dict()
        with open(os.path.join(config.output_dir, "state_dict.json"), "w") as f:
            json.dump(state_dict, f)
        

if __name__ == "__main__":
    main()
