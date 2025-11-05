import urllib.parse
import json
import os
base_path = os.path.dirname(os.path.abspath(__file__))


def load_file(path):
    path = os.path.join(base_path, path)
    with open(path) as f:
        for l in f:
            data = json.loads(l)
            key = data.pop('key')
            yield key, data



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
