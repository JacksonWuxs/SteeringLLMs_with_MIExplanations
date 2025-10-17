import time
import re
import string
import json
import math
import concurrent
import collections
import multiprocessing

import tqdm


from OpenaiAPI import Chatting


class TextSpanExplainer:
    def __init__(self, key):
        instruct = "You are studying a neural network. Each neuron looks for one particular concept/topic/theme/behavior/pattern. " +\
               "Look at some words the neuron activates for and guess what the neuron is looking for. " +\
               "Note that, some neurons may only look for a particular text pattern, while some others may be interested in very abstractive concepts. " +\
               "Pay more attention to the words in the front as they supposed to be more correlated to the neuron behavior. " +\
               "Don't list examples of words and keep your summary as detail as possible. " +\
               "If you cannot summarize most of the words, you should say ``Cannot Tell.``"
        examples = [("accommodation, racial, ethnic, discrimination, equality, apart, utterly, legally, separately, holding, implicit, unfair, tone.",
                 "Social justic and discrimination."),
                ("B., M., e., R., C., OK., A., H., D., S., J., al., p., T., N., W., G., a.C., or, St., K., a.m., L..",
                 "Cannot Tell."),
                ("Scent, smelled, flick, precious, charm, brushed, sealed, smell, brace, curios, sacred, variation, jewelry, seated.",
                 "Perception of scents and precious objects."),
                ("BP, HR, RR, O2 Sat, T, Ht, UO, BMI, BSA.",
                 "Medical measurements in emergency rooms."),
                ("actual, literal, real, Really, optical, Physical, REAL, virtual, visual.",
                 "Perception of reality."),
                ("Go, Python, Java, c++, python3, c#, java, Ruby, Swift, PHP.",
                 "Morden programming language."),
                ("1939-1945, 1945, 1942, 1939, 1940, 1941.",
                 "Years of the second world war."),
                ("1976, 1994, 1923, 2018, 2014, 1876, 1840.",
                 "Cannot Tell."),
                ]
        self.model = Chatting.GPT4oMini(KEY, cache=False,
                                   system=instruct, examples=examples,
                                   temperature=0.0001, top_p=0.0001, n=1)

    def __call__(self, cases):
        if isinstance(cases, str):
            cases = [cases]
        if not isinstance(cases, (tuple, list)):
            cases = list(cases)
        cases = map(self.format, cases)
        return list(map(self.clean, self.model.batch_call(cases)))

    def format(self, case):
        return case

    def clean(self, case):
        temp = set(_ for _ in case if "cannot tell" not in _.lower())
        if len(temp) == 0:
            return "Cannot Tell."
        return " or ".join(_.split(".")[0] for _ in temp) + '.'


class TextSpanJudge:
    def __init__(self, key):
        instruct = "You are a linguistic expert. " +\
                   "Provide a short analysis on whether the words well represent the given concept/topic/theme/pattern. " +\
                   "Organize your final decision in the format of ``Final Decision: [[ Yes/Probably/Unlikely/No ]]``."
        self.model = Chatting.GPT4oMini(KEY, system=instruct, examples=None, cache=False,
                                   temperature=0.0001, top_p=0.0001, n=3)

    def __call__(self, cases):
        cases = map(self.format, cases)
        cases = self.model.batch_call(cases)
        return list(map(self.clean, cases))

    def format(self, case):
        case[1] = case[1].replace("\\n", "\n").strip()
        #case[1] = "\nSpan".join(case[1].split("\nSpan")[:4])
        return "Concept/Topic/Theme/Pattern: %s.\nWords: %s" % tuple(case)

    def clean(self, verify):
        verifies = collections.Counter()
        for verify in verify:
            verify = verify.lower().split("decision")[-1]
            if "[[" in verify and "]]" in verify:
                verify = verify.split("[[", 1)[-1].rsplit("]]", 1)[0].strip()
            if verify.startswith(": "):
                verify = verify[2:].split(".", 1)[0].strip()
            verifies[verify] += 1
        return verifies.most_common(1)[0][0]
    

if __name__ == "__main__":
    KEY = "XXXXXXXXXXXXXXXX"
    
    model = TextSpanExplainer(KEY)
    judge = TextSpanJudge(KEY)

    import sys
    file = sys.argv[1]
    print("Annotating File: %s" % file)
    with open(file, encoding="utf8") as f:
        headline = f.readline().strip().split("\t")
        idx = headline.index("FeatureID")
        text = headline.index("Words")
        fullset = [(_.split("\t")[idx], ", ".join(_.split("\t")[text].split("|||")[:15]) + '.') for _ in f.read().strip('\n').split("\n")]
    results = [[x[0], "-", "Cannot Tell.", x[1]] for x in fullset]#[:100]
    
    need_explain = [item for item in results if len(item[3]) > 0]
    explanations = model(_[3] for _ in need_explain)
    for item, expl in zip(need_explain, explanations):
        item[1], item[2] = 'no', expl
        
    need_verify = [item for item in results if "cannot tell" not in item[2].lower()]
    verifications = judge([_[2], _[3]] for _ in need_verify)
    for item, verify in zip(need_verify, verifications):
        item[1] = verify


    from collections import Counter
    c = Counter([_[1] for _ in results])
    print("Explainability: %.4f" % ((c["yes"] + c["probably"]) / sum(c.values())))
    for cate, freq in c.items():
        print(cate, freq / len(results))
    with open(file.rsplit(".", 1)[0] + "_explained.tsv", "w", encoding="utf8") as f:
        f.write("FeatureID\tVerify\tSummary\tWords\n")
        for idx, verify, summary, words in results:
            words = words.replace("\n", "\\n").replace("\t", "\\t").replace("\r", "")
            summary = summary.replace("\n", "\\n").replace("\t", "\\t").replace("\r", "")
            verify = verify.replace("\n", "\\n").replace("\t", "\\t").replace("\r", "")
            f.write("%s\t%s\t%s\t%s\n" % (idx, verify, summary, words))


