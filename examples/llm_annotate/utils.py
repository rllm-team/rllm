import sys
import os.path as osp
import random
import json

import torch

sys.path.append("./")
sys.path.append("../")

from rllm.llm import Predictor
from rllm.llm.llm_module.langchain_llm import LangChainLLM


def annotate(name, label_names, target_table, mask, llm, use_cache=False):
    """
    Annotate selected samples using LLM Predictor.
    """
    print(f"Annotating {mask.sum()} samples")

    pred = -1 * torch.ones(target_table.df.shape[0], dtype=torch.long)

    if use_cache:
        cached_annotation = load_cache(name)
        if cached_annotation:
            pred = torch.tensor(cached_annotation['pred'])
            cached_mask = torch.tensor(cached_annotation['mask'])
            hit_num = (mask & cached_mask).sum()
            print(f'cache hit num: {hit_num}')
            mask = mask & ~cached_mask
        else:
            cached_mask = mask & ~mask
    else:
        cached_mask = mask & ~mask

    if mask.sum() == 0:
        print("All nodes already annotated.")
        return pred

    df = target_table.df.loc[mask.cpu().numpy()].drop(columns=[target_table.target_col])

    if name == "tlf2k":
        scenario = "Classify the artists into one of the given labels."
        df["biography"] = df["biography"].str[:1000]
    elif name == "tacm12k":
        scenario = "Classify the papers into one of the given conferences. The descriptions to the conferences are as follows: CIKM (Conference on Information and Knowledge Management): Focuses on research at the intersection of information retrieval, data management, and knowledge discovery. \nCOLT (Conference on Learning Theory): Dedicated to theoretical aspects of machine learning and statistical learning theory. \nICML (International Conference on Machine Learning): One of the top conferences for presenting cutting-edge research in machine learning. \nKDD (Knowledge Discovery and Data Mining): Premier conference for research on data mining, data science, and big data analytics. \nMobiCom (International Conference on Mobile Computing and Networking): Covers research on mobile systems, wireless networks, and mobile computing technologies. \nSIGCOMM (ACM Conference on Applications, Technologies, Architectures, and Protocols for Computer Communication): Leading venue for research in computer networking and communication systems. \nSIGIR (Special Interest Group on Information Retrieval): Premier forum for presenting research on information retrieval and search technologies. \nSIGMOD (Special Interest Group on Management of Data): Focuses on database systems and data management technologies. \nSODA (Symposium on Discrete Algorithms): Covers theoretical and practical aspects of algorithms and discrete mathematics. \nSOSP (Symposium on Operating Systems Principles): Top conference for innovations in operating systems and distributed systems. \nSPAA (Symposium on Parallelism in Algorithms and Architectures): Focuses on parallel computing in both theoretical and practical aspects. \nSTOC (Symposium on Theory of Computing): A flagship conference for theoretical computer science research. \nVLDB (Very Large Data Bases Conference): Leading venue for data management and large-scale data systems. \nWWW (The Web Conference): Covers web-related research including web mining, information retrieval, and web applications.\n"
    elif name == "tml1m":
        scenario = "Classify the users into one of the given age ranges.They denote the lower bound of age ranges. The meaning of occupation code is shown below \n 0: other or not specified 1: academic/educator 2: artist 3: clerical/admin 4: college/grad student 5: customer service 6: doctor/health care 7: executive/managerial 8: farmer 9: homemaker 10: K-12 student 11: lawyer 12: programmer 13: retired 14: sales/marketing 15: scientist 16: self-employed 17: technician/engineer 18: tradesman/craftsman 19: unemployed 20: writer. For example, 1 denotes 1-17 years old and 18 denotes 18-24 years old. You should only reply with one of those given numbers. The dataset contains patterns where the Occupation feature is a key determinant of the class value. When classifying new data points, pay close attention to the Occupation feature, as it provides actionable information for assigning the correct class. For example, if a data point has an Occupation value of 10, it is most likely associated with class 1. Similarly, an Occupation value of 4 corresponds to class 18, while an Occupation value of 17 aligns with class 25. Data points with an Occupation value of 7 can belong to either class 35 or class 56, indicating some overlap or ambiguity in this specific occupation category. Additionally, an Occupation value of 0 maps to class 45, and an Occupation value of 1 corresponds to class 50. These patterns suggest that the Occupation feature is highly predictive of the class, and decision-makers should prioritize this feature when classifying new data points. By identifying the Occupation value of a new data point and matching it to the prototypical examples provided, one can reliably assign the appropriate class label."
    else:
        scenario = "Classify the given text into one of the given categories"
    labels = ", ".join(label_names)

    predictor = Predictor(llm=LangChainLLM(llm), type="classification")
    outputs = predictor(df, scenario=scenario, labels=labels)

    select_pred = []
    for output in outputs:
        output = output.lower()
        matches = []
        for label in label_names:
            if label.lower() in output.lower():
                matches.append(label)
        if matches:
            matched = max(matches, key=len)
        else:
            matched = random.choice(label_names)
        select_pred.append(label_names.index(matched))

    select_pred = torch.tensor(select_pred)
    pred[mask] = select_pred

    if use_cache:
        save_cache(name, pred.tolist(), (mask | cached_mask).tolist())
        print(f"{mask.sum()} updated to cache")

    return pred


def save_cache(name, pred, mask):
    with open('cache/' + name + '_pl_cache.json', "w") as f:
        json.dump({'pred': pred, 'mask': mask}, f, indent=4)


def load_cache(name):
    name = name.lower()
    cache_path = 'cache/' + name + '_pl_cache.json'
    if osp.exists(cache_path):
        with open(cache_path, 'r') as f:
            return json.load(f)
    return None