#!/bin/python3
import os
import sys
sys.path.append('.')

from LLMRouter import datasets
import click

import json
import pickle
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split


@click.command()
@click.option("--dataset", prompt="dataset name(gsm8k_train)")
@click.option("--models", prompt="llm model")
@click.option("--output", prompt="output path")
def run(dataset, models, output):
    # Threshold for correctness
    THRESHOLD = 0.8

    model_names = models.split(',')
    records = datasets.load_dataset([dataset], model_names, ["similar"])
    OUTPUT_PATH = output

    n_samples = len(records)
    n_models = len(model_names)
    # Prepare arrays
    prompts = []
    scores = np.zeros((n_samples, n_models))
    tokens = np.zeros((n_samples, n_models))
    times = np.zeros((n_samples, n_models))

    print("\nProcessing data...")
    for i, model_data in enumerate(records):
        question = model_data['question']
        prompts.append(question)
        item_key = model_data.get('key', f'item_{i}')

        for j, model_name in enumerate(model_names):
            # Get response data
            if model_data.get(model_name+'_response', None):
                response = model_data[model_name+'_response']
                tokens[i, j] = response['usage']['total_tokens']
                times[i, j] = response['response_time']
            else:
                print(f"Warning: Missing response for {item_key} from {model_name}")
                tokens[i, j] = 0
                times[i, j] = 0

            # Get eval score
            if model_data.get(model_name+'_similar_eval', None):
                eval_point = model_data[model_name+'_similar_eval']['point']
                # Threshold: >= 0.9 is correct (1), otherwise incorrect (0)
                scores[i, j] = 1.0 if eval_point >= THRESHOLD else 0.0
            else:
                # If no eval, mark as 0
                scores[i, j] = 0.0

    # Split into train/val/test (60/20/20)
    indices = np.arange(n_samples)
    train_idx, temp_idx = train_test_split(indices, test_size=0.4, random_state=42)
    val_idx, test_idx = train_test_split(temp_idx, test_size=0.5, random_state=42)

    print(f"\nData split:")
    print(f"  Train: {len(train_idx)}")
    print(f"  Val: {len(val_idx)}")
    print(f"  Test: {len(test_idx)}")

    # Create router dataset
    router_dataset = {
        'prompt': {
            'train_prompt': [prompts[i] for i in train_idx],
            'val_prompt': [prompts[i] for i in val_idx],
            'test_prompt': [prompts[i] for i in test_idx]
        },
        'data': {
            'train_score': scores[train_idx],
            'val_score': scores[val_idx],
            'test_score': scores[test_idx],
            'test_tokens': tokens[test_idx],
            'test_time': times[test_idx]
        },
        'model': model_names
    }

    # Print statistics
    print("\n" + "="*75)
    print("Dataset Statistics:")
    print("="*75)
    for j, model_name in enumerate(model_names):
        train_acc = np.mean(scores[train_idx, j])
        val_acc = np.mean(scores[val_idx, j])
        test_acc = np.mean(scores[test_idx, j])
        avg_tokens = np.mean(tokens[test_idx, j])
        avg_time = np.mean(times[test_idx, j])

        print(f"\n{model_name}:")
        print(f"  Train Acc: {train_acc:.4f}")
        print(f"  Val Acc:   {val_acc:.4f}")
        print(f"  Test Acc:  {test_acc:.4f}")
        print(f"  Avg Tokens: {avg_tokens:.2f}")
        print(f"  Avg Time:   {avg_time:.4f}s")

    # Save to pickle
    output_path = Path(OUTPUT_PATH)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'wb') as f:
        pickle.dump(router_dataset, f)

    print(f"\n✓ Dataset saved to {OUTPUT_PATH}")
    print("="*75)




if __name__ == '__main__':
    run()
