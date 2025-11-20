#!/bin/python3
import os
import sys
sys.path.append('.')

from LLMRouter import datasets
import click

@click.command()
@click.option("--dataset", prompt="dataset name(gsm8k_train)")
@click.option("--model", prompt="llm model")
def run(dataset, model):
    return datasets.add_model_response(dataset, model)


if __name__ == '__main__':
    run()
