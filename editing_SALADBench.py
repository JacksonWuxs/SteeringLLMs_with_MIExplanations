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



def generate_answers(sae, generator, name="mistral", sampling=500):
    root = "outputs/benchmarks/%s.jsonl" % name
    print("Loading Data From: %s" % root)
    if True:#not os.path.exists(root): 
        data = datasets.load_dataset("OpenSafetyLab/Salad-Data", 
                                 name="attack_enhanced_set", 
                                 split="train").shuffle(0)
        
        
        with open(root, "w", encoding='utf8') as f:
            n_row = 0
            if f.tell() != 0:
                f.seek(0)
                for n_row, _ in enumerate(f, 1):
                    pass
            if n_row < sampling:
                try:
                    for idx in tqdm.trange(n_row, len(data), desc="Generate"):
                        q = data[idx]["augq"]
                        p = "<s>[INST] %s [/INST]" % q
                        sae.gen_step = 0
                        a = generator.generate(p,
                                   do_sample=False,
                                   max_new_tokens=512)
#                        print('\n\n', idx, a)
                        f.write(json.dumps({"Q": q, "A": a, "O": data[idx]["baseq"]}) + '\n')
                        f.flush()
                        if idx == sampling:
                            break
                except KeyboardInterrupt:
                    pass
                
    Q, A = [], []
    with open(root, "r", encoding="utf8") as f:
        for row in f:
            row = json.loads(row.strip())
            Q.append(row.get("O", row["Q"]))
            A.append(row["A"])
            if len(Q) == sampling:
                break
    return Q, A


def evaluate_safety(Q, A):
    md_judge = MDJudgeEvaluator("OpenSafetyLab/MD-Judge-v0.1", #trust_remote_code=True,
                               use_vllm=False, use_flash_attn=False)
    eval_labels, eval_clses = md_judge.eval_unsafe_labels(Q, A)
    return eval_labels.count(1) / (eval_labels.count(0) + eval_labels.count(1))




HARMFUL = {"s1": "violent crimes", "s2": "non-violent crimes",
           "s3": "sex-related crimes", "s4": "child sexual exploitation",
           "s5": "defamation", "s6": "specialized advice",
           "s7": "privacy", "s8": "intellectual property",
           "s9": "indiscriminate weapons", "s10": "hate",
           "s11": "suicide & self-harm", "s12": "sextual content",
           "s13": "elections", "s14": "code interpreter abuse"}
HARMLESS = {"d1": "physical", "d2": "cyber",
            "d3": "legal", "d4": "financial",
            "d5": "privacy", "d6": "ethical",
            "d7": "victim support"}

def load_annotations(root, subset, check):
    counts = collections.Counter()
    masks = []
    with open(root, "r", encoding="utf8") as f:
        f.readline()
        for ttl, row in enumerate(f, 1):
            row = row.strip().split("\t")
            if row[1].strip().split("|||")[0] == "none":
                continue
            if row[2].strip().split("|||")[0] not in check:
                continue
            counts.update(_.strip() for _ in row[1].split("|||")[0].strip().split(","))
            masks.append(int(row[0]))
    print("Totally masking %d/%d features." % (len(masks), ttl))
    for key, val in subset.items():
        print(key, counts[key] / ttl, val)
    return masks



if __name__ == "__main__":
    import random
    sys.argv.append("./outputs/weights/TopK7_l8_h65k_epoch5.pth")
    sys.argv.append("./outputs/annotations/MI_TopK7_l8_h65k_epoch5_MIv2_wordlist_explained_defense_v1.tsv")
    sys.argv.append("./outputs/annotations/MI_TopK7_l8_h65k_epoch5_MIv2_wordlist_explained_defense_v1.tsv")
    print(sys.argv)
    name, layer, sae = load_pretrained(sys.argv[2])
    generator = Generator("mistralai/Mistral-7B-Instruct-v0.2", device="cuda")
    mount_function(generator._model, "mistral", int(layer), sae)
    switch_mode(sae, "edit")
    mask1 = set(load_annotations(sys.argv[3], HARMFUL, check={"yes",}))
    mask2 = set(load_annotations(sys.argv[4], HARMLESS, check={"yes", }))

    sae.enforce_actv(mask2, magnitude=2.5)
    #sae.set_edit(mask1, "turnoff", mag=1.0)

    Q, A = generate_answers(sae, generator, name="final_l8_defenseV2_a25_%s" % name)
    print("ASR=%.4f" % evaluate_safety(Q, A))

