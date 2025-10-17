import os
import sys
import collections
import json

from generator import Generator
from llm_surgery import switch_mode, mount_function
from autoencoder import load_pretrained

import tqdm
import torch as tc
import numpy as np
import transformers as trf
import datasets

from saladbench.evaluator import MDJudgeEvaluator



def generate_answers(root, sampling):
    O, Q, A = [], [], []
    data = datasets.load_dataset("OpenSafetyLab/Salad-Data",
                                 name="attack_enhanced_set",
                                 split="train").shuffle(0)
    with open(root, "r", encoding="utf8") as f:
        for i, row in enumerate(f):
            row = json.loads(row.strip())
            O.append(data[i]["baseq"])
            Q.append(row["Q"])
            A.append(row["A"])
            if len(Q) == sampling:
                break
    return O, Q, A


def evaluate_safety(O, Q, A):
    md_judge = MDJudgeEvaluator("OpenSafetyLab/MD-Judge-v0.1", #trust_remote_code=True,
                               use_vllm=False, use_flash_attn=False)
    eval_labels, eval_clses = md_judge.eval_unsafe_labels(O, A)

    for i in [100, 200, 300, 400, 500]:
        print(i, '%.4f' % (eval_labels[:i].count(1) / (eval_labels[:i].count(0) + eval_labels[:i].count(1))))
    return eval_labels.count(1) / (eval_labels.count(0) + eval_labels.count(1))




if __name__ == "__main__":
    maximum  = None if len(sys.argv) <= 3 else int(sys.argv[3])
    O, Q, A = generate_answers(sys.argv[2], maximum)
    print("ASR=%.4f" % evaluate_safety(O, Q, A))

