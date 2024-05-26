# This source code is licensed under the MIT license

import json
import pdb
import pandas as pd
from tabulate import tabulate
import matplotlib.pyplot as plt

from utils import get_final_result_gsm8k, get_final_result_math, get_csqa_match

models = ['FILM-7B', 'Mistral-7B-Instruct-v0.2']
tasks = ['gsm8k_8shot', 'math_4shot', 'csqa_0shot']

info = {}
for model in models:
    dict_tmp = {}
    for task in tasks:

        file_label = '/content/FILM/short_tasks/prompts/{task}.jsonl'.format(task=task)
        file_pred = '/content/FILM/short_tasks/results/{model}/sample_{task}.jsonl'.format(model=model, task=task)

        acc_list = []
        with open(file_label, 'r', encoding='utf-8') as f_label, \
                open(file_pred, 'r', encoding='utf-8') as f_pred:
            label_infos = [json.loads(line) for line in f_label.readlines()]
            pred_infos = [json.loads(line) for line in f_pred.readlines()]

            assert len(label_infos) == len(pred_infos)

            for label_info, pred_info in zip(label_infos, pred_infos):

                if 'gsm8k' in task:
                    pred = pred_info['samples'][0]
                    label = label_info['completion']

                    pred_result = get_final_result_gsm8k(pred)
                    label_result = get_final_result_gsm8k(label)

                    assert label_result is not None
                    assert label_result != ''
                    assert label_result != 0

                    if pred_result == label_result:
                        acc_list.append(1)
                    else:
                        acc_list.append(0)

                elif 'math' in task:
                    pred = pred_info['samples'][0]
                    label = label_info['completion']

                    pred_result = get_final_result_math(pred)
                    label_result = get_final_result_math(label)

                    if ',' not in label_result:
                        pred_result = pred_result.replace(',', '')

                    assert label_result is not None
                    assert label_result != ''
                    assert label_result != 0

                    if pred_result == label_result:
                        acc_list.append(1)
                    else:
                        acc_list.append(0)

                elif 'csqa' in task:
                    pred = pred_info['samples'][0]
                    label = label_info['answer']
                    candidates = label_info['candidates']

                    assert label in candidates

                    score = get_csqa_match(pred, label, candidates)
                    acc_list.append(score)

                else:
                    pdb.set_trace()

        
        dict_tmp[task] = sum(acc_list) / len(acc_list)
    info[model] = dict_tmp

df = pd.DataFrame(info)
df = df.transpose()

for dataset, scores in df.items():
    fig, ax = plt.subplots(figsize=(10, 6))

    for model, score in scores.items():
        ax.bar(model, score, label=model)

    ax.set_xlabel('Models')
    ax.set_ylabel('Scores')
    ax.set_title(f'Scores en {dataset}')
    ax.legend()

    plt.xticks(rotation=45, ha='right')

    plt.tight_layout()
    plt.show()