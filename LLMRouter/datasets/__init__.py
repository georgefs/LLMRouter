import urllib.parse
import json
import os
from openai import OpenAI
from datetime import datetime
import time
base_path = os.path.dirname(os.path.abspath(__file__))


def real_path(path):
    path = os.path.join(base_path, path)
    return path


def load_file(path):
    path = real_path(path)
    if not os.path.exists(path):
        return
    with open(path) as f:
        for l in f:
            data = json.loads(l)
            key = data.pop('key')
            yield key, data


model_mapping = {
    "openai/gpt-oss-20b": ["openai/gpt-oss-20b:free"],
}


def add_model_response(dataset, model):
    dataset_path = f"datasets/{dataset}.jsonl"
    model_filename = urllib.parse.quote_plus(model)
    response_path = f"responses/{dataset}/{model_filename}.jsonl"
    qa_dataset = load_file(dataset_path)
    response_dataset = dict([v for v in load_file(response_path)])

    # default use openrouter
    base_url = os.environ.get('OPENAI_API_BASEURL')
    api_key = os.environ.get('OPENAI_API_KEY')

    client = OpenAI(
      base_url=base_url,
      api_key=api_key,
    )
    target_model = model_mapping[model][0]

    response_tmp_path = response_path + ".tmp"
    with open(real_path(response_tmp_path), "w+") as response_f:
        for key, data in qa_dataset:
            print(key)
            if key in response_dataset:
                response_data = response_dataset[key]
                response_data['key'] = key
            else:
                messages=[
                      {
                          "role": "user",
                          "content": data['question']
                          }
                      ]
                st = datetime.now()
                completion = client.chat.completions.create(
                  extra_body={},
                  model=target_model,
                  messages=messages
                )
                ed = datetime.now()
                dt = (ed - st).total_seconds()
                response_data = {
                    "key": key,
                    "text": completion.choices[0].message.content,
                    "usage": {
                        "input_tokens": completion.usage.prompt_tokens, 
                        "output_tokens": completion.usage.completion_tokens,
                        "total_tokens": completion.usage.total_tokens
                    },
                    "response_time": dt
                }
                print(response_data)
                response_dataset[key] = response_data

            response_f.write(json.dumps(response_data)+"\n")

    os.replace(real_path(response_tmp_path), real_path(response_path))






def load_dataset(datasets, models, eval_methods):
    for dataset in datasets:
        dataset_path = f"datasets/{dataset}.jsonl"
        result = dict([v for v in load_file(dataset_path)])

        for model in models:
            model_filename = urllib.parse.quote_plus(model)
            model_response_path = f"responses/{dataset}/{model_filename}.jsonl"
            responses_data = load_file(model_response_path)
            for key, response in responses_data:
                result[key][f'{model}_response'] = response


            for eval_method in eval_methods:
                model_eval_path = f"evals/{dataset}/{eval_method}/{model_filename}.jsonl"
                eval_datas = load_file(model_eval_path)
                for key, eval_data in eval_datas:
                    result[key][f'{model}_{eval_method}_eval'] = eval_data

        
    for key, r in result.items():
        r['key'] = key

    return [v for v in result.values()]


if __name__ == '__main__':
    dataset = load_dataset(["gsm8k"], ["mistralai/Mixtral-8x7B-Instruct-v0.1" ,"gpt-4-1106-preview"], ["similar"])
    for l in dataset:
        print(l)
