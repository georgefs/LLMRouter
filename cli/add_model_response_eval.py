#!/bin/python3
import os
import sys
sys.path.append('.')

from LLMRouter import datasets
import click

@click.command()
@click.option("--dataset", prompt="dataset name(gsm8k_train)")
@click.option("--model", prompt="llm model(Llama-3.1-70B, Llama-3.1-8B-Instruct, Llama-3.1-TAIDE-LX-8B-Chat,  Llama-4-Scout-17B-16E-Instruct-FP8, gpt-oss-20b)")
@click.option("--eval_method", prompt="eval method(random, similar)")
def run(dataset, model, eval_method):
    return datasets.add_model_response_eval(dataset, model, eval_method)


if __name__ == '__main__':
    run()
