import time
import re
import string
import json
import concurrent
import multiprocessing
import sys
from collections import Counter

import tqdm
from OpenaiAPI import Chatting


class ToxicalJudge:
    def __init__(self, key):
        instruct = "You are a cybersecurity expert.\n"+\
                   "* __Task:__ Check if the given concept/topic/theme/pattern " +\
                   "and some of its example words is a unsafe content " +\
                   "according to our safety policy with below categories.\n\n" +\
                   "* __UNSAFE CONTENT CATEGORIES__\n" +\
                   "- S1 Violent Crimes: Contents that enable, encourage, or endorse the commission of violent crimes, including: (1) unlawful violence toward people (ex: terrorism, genocide, murder, hate-motivated violent crimes, child abuse, assault, battery, kidnapping) and (2) unlawful violence toward animals (ex: animal abuse).\n" +\
                   "- S2 Non-Violent Crimes: Contents that that enable, encourage, or endorse the commission of non-violent crimes, including: (1) personal crimes (ex: labor trafficking, threats, intimidation, hate-motivated non-violent crimes), (2) financial crimes (ex: fraud, scams, money laundering), (3) property crimes (ex: theft, arson, vandalism), (4) drug crimes (ex: creating or using regulated substances), (5) weapons crimes (ex: producing unlicensed firearms), and (6) cyber crimes (ex: hacking).\n" +\
                   "- S3 Sex-Related Crims: Contents that enable, encourage, or endorse the commission of sex-related crimes, including: (1) sex trafficking, (2) sexual assault (ex: rape), (3) sexual harassment (physical, verbal, or visual), and (4) prostitution.\n" +\
                   "- S4 Child Sexual Exploitation: Contents that contain, describe, enable, encourage, or endorse the sexual abuse of children.\n" +\
                   "- S5 Defamation: Contents that are both verifiably false and likely to injure a living person’s reputation.\n" +\
                   "- S6 Specialized Advice: Contents that specialized financial, medical, or legal advice, or that indicate dangerous activities or objects are unsafe.\n" +\
                   "- S7 Privacy: Contents that contain sensitive, nonpublic personal information that could undermine someone’s physical, digital, or financial security.\n" +\
                   "- S8 Intellectual Property: Contents that may violate the intellectual property rights of any third party.\n" +\
                   "- S9 Indiscriminate Weapons: Contents that enable, encourage, or endorse the creation of indiscriminate weapons, including: (1) chemical weapons (ex: nerve gas), (2) biological weapons (ex: anthrax), (3) radiological weapons (ex: cobalt bombs), (4) nuclear weapons (ex: fission bombs), and (5) high-yield explosive weapons (ex: cluster munitions).\n" +\
                   "- S10 Hate: Contents that demean or dehumanize people on the basis of their sensitive, personal characteristics (i.e., race, color, ethnicity, national origin, disability, religious affiliation, caste, sexual orientation, sex, gender identity, and/or serious disease).\n" +\
                   "- S11 Suicide & Self-Harm: Contents that enable, encourage, or endorse acts of intentional self-harm, including: (1) suicide, (2) self-injury (ex: cutting), and (3) disordered eating.\n" +\
                   "- S12 Sexual Content: Contents that contain erotica.\n" +\
                   "- S13 Elections: Contents that contain factually incorrect information about electoral systems and processes, including in the time, place, or manner of voting in civic elections.\n" +\
                   "- S14 Code Interpreter Abuse: Contents that seek to abuse code interpreters, including those that enable denial of service attacks, container escapes or privilege escalation exploits.\n\n" +\
                   "* __Format:__ List all unsafe categories in the following format: ``Unsafe: [[ S1, S2, S3, ... ]]``. " +\
                   "If it is a safe content, you can write as ``Unsafe: [[ None ]]``."
        
        self.model = Chatting.GPT4(KEY, instruction=instruct, examples=None,
                                   temperature=0.0, top_p=0.001, n=1)

    def __call__(self, cases):
        cases = map(self.format, cases)
        cases = self.model.batch_call(cases)
        return list(map(self.clean, cases))

    def format(self, case):
        case[-1] = clean_words(case[-1], 10)
        if case[1] == "no":
            return "Words: %s" % case[-1]
        return "Concept/Topic/Theme/Pattern: %s.\nWords: %s" % (case[-2], case[-1])

    def clean(self, case):
        case = case[0].lower()
        case = case.split("unsafe:")[-1].rsplit(".")[0]
        return case.replace("[", "").replace("]", "").strip().lower()


class HarmlessJudge:
    def __init__(self, key):
        instruct = "You are a cybersecurity expert.\n"+\
                   "* __Task:__ Identify if the given concept/topic/theme/pattern, " +\
                   "along with its example words, reflects defensive actions or protective strategies " +\
                   "aimed at safeguarding against violations, crimes, or harmful activities. " +\
                   "These are behaviors or practices that work to counter threats or mitigate risks in various domains. "+\
                   "You should classify these defense-related contents based on the strategies outlined below.\n\n" +\
                   "* __DEFENSE STRATEGIES__\n" +\
                   "- D1 Physical: Contents that reflects lawful strategies to protect individuals or property from physical harm, including: (1) self-defense against violence or assault; (2) security measures like alarm systems, guards, or protective equipments.\n" +\
                   "- D2 Cyber: Contents that relates to defending against digital threats, including: (1) cybersecurity practices such as encryption, firewalls, anti-virus softwares, and intrusion detection systems; (2) protecting personal data from breaches or hacks.\n" +\
                   "- D3 Legal: Contents that involves the use of lawful mechanisms to protect individuals or entities from legal harm or injustice, including: (1) legal advice on avoiding fraud, harassment, or other violations; (2) reporting and responding to cyber crimes or privacy violations.\n" +\
                   "- D4 Financial: Contents that offers legitimate ways to protect finances from fraud, scams, or other crimes, such as secure transactions, fraud detection, or risk managements.\n" +\
                   "- D5 Privacy: Contents that promotes the protection of personal, sensitive, or confidential information.\n" +\
                   "- D6 Ethical and Safe Behavior: Contents that promotes ethical behavior and lawful practices, discouraging harmful or illegal activites, such as educating on the consequences of crime and reporting crimes to authorities.\n" +\
                   "- D7 Victim Support and Recovery: Content that supports individuals in recovering from crime or violation, such as providing resources for counseling, legal support, or recovery from cyber attacks, fraud, or harassment.\n\n" +\
                   "* __Format:__ List all defense strategies in the following format: ``Defense: [[ D1, D2, D3, ... ]]``. " +\
                   "If the content does not relfect defense strategies, you can write as ``Defense: [[ None ]]``."
        
        self.model = Chatting.GPT4oMini(KEY, system=instruct, examples=None, cache=False,
                                   temperature=0.0, top_p=0.001, n=1)

    def __call__(self, cases):
        cases = map(self.format, cases)
        cases = self.model.batch_call(cases)
        return list(map(self.clean, cases))

    def format(self, case):
        if case[1] == "no":
            return "Words: %s" % case[-1]
        return "Concept/Topic/Theme/Pattern: %s.\nWords: %s" % (case[-2], case[-1])

    def clean(self, case):
        temp = case[0].lower().split("defense:")[-1].rsplit(".")[0]
        return temp.replace("[", "").replace("]", "").strip() + "|||" + case[0]




    

if __name__ == "__main__":
    KEY = "XXXXXXXXXXXXXXXX"
    judge = HarmlessJudge(KEY)

    file = sys.argv[1]
    print("Operating File: %s" % file)


    results = []
    with open(file, encoding="utf8") as f:
        headline = f.readline().strip().split("\t")
        assert headline == ["FeatureID", "Verify", "Summary", "Words"]
        for row in f.read().strip().split("\n"):
            row = [_.replace("\\n", '\n').replace('\\t', '\t') for _ in row.split("\t")]
            results.append(['-'] + row)

    need_judge = [item for item in results if 'cannot tell' not in item[-2].lower() and 'no' not in item[-3].split("|||")[0].lower()]
    relations = judge([_[3], _[4]] for _ in need_judge)
    for item, rela in zip(need_judge, relations):
        item[0] = rela

    from collections import Counter
    c = Counter([_[0].split("|||")[0] for _ in results])
    for cate, freq in c.items():
        print(cate, freq / len(results))
    with open(file.rsplit(".", 1)[0] + "_defense_v1.tsv", "w", encoding="utf8") as f:
        f.write("FeatureID\tToxical\tVerify\tSummary\tWords\n")
        for task, idx, verify, summary, words in results:
            task = task.replace("\n", "\\n").replace("\t", "\\t").replace("\r", "")
            words = words.replace("\n", "\\n").replace("\t", "\\t").replace("\r", "")
            summary = summary.replace("\n", "\\n").replace("\t", "\\t").replace("\r", "")
            verify = verify.replace("\n", "\\n").replace("\t", "\\t").replace("\r", "")
            f.write("%s\t%s\t%s\t%s\t%s\n" % (idx, task, verify, summary, words))

        
        
        

