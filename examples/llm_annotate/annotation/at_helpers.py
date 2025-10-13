import ast
import re
from collections import Counter

import torch


def get_final_results(res, label_names, select_ids, gt):
    """
    Generate final predictions and confidences from retrieved answers.
    """
    pred = []
    gt_y = gt[select_ids]
    conf = []
    cannot_fix = 0
    for i, r in enumerate(res):
        if i not in select_ids: continue
        k = len(r)
        this_pred = []
        this_conf = []
        selected = False
        for selection in r:

            if selection[0][0] not in label_names:
                continue
            p = label_names.index(selection[0][0])
            c = selection[0][1]
            this_pred.append(p)
            this_conf.append(c)
            selected = True
        if not selected:
            cannot_fix += 1
            p = get_closest_label(selection[0][0], label_names)
            c = selection[0][1] / 2
            p = label_names.index(p)
            pred.append(p)
            conf.append(c)
            continue
        counter = Counter(this_pred)
        most_common = counter.most_common(1)
        p = most_common[0][0]
        first_appear = this_pred.index(p)
        base_c = 0
        orig_c = this_conf[first_appear]

        for pp, cc in zip(this_pred, this_conf):
            if pp == p:
                this_c = (orig_c + cc) / 2
                base_c += this_c
            else:
                base_c += (100 - cc)
        base_c /= k
        pred.append(p)
        conf.append(base_c)

    pred = torch.tensor(pred)
    conf = torch.tensor(conf) / 100.0
    all_acc = (pred == gt_y).float().mean()
    filter_acc = (pred[conf > 0] == gt_y[conf > 0]).float().mean()
    filter_label = gt_y[conf > 0]
    print("cannot fix number: {}".format(cannot_fix))
    print("all acc: {:.2f}, filter acc: {:.2f}, number of labels: {}".format(all_acc, filter_acc, len(filter_label)))

    return pred, conf


def generate_prompts(name, texts, select_idx, label_names):
    """
    Generate prompt list based on raw text.
    """
    prompts = ["" for _ in range(len(texts))]
    for i, text in enumerate(texts):
        if i in select_idx:
            prompts[i] = topk_prompt(name, text, label_names)

    return prompts


def query_llm(llm, prompts, label_names, n=1):
    """
    Query LLM n times for each prompt.
    """
    print('Start annotation.')
    responses = []
    for i, prompt in enumerate(prompts):
        if prompt == '':
            responses.append('')
            continue

        res = []
        for j in range(n):
            print('querying', i, j)
            output = llm.invoke(prompt)
            print('response:', output)
            ok, error_type = validate_syntax(output, label_names)
            if not ok:
                print('correcting', i, j)
                correction_prompt = generate_correction_prompt(prompt, label_names, error_type)
                output = llm.invoke(correction_prompt)
                print('response:', output)
            res.append(output)
        responses.append(res)

    return responses


def collect_answers(answer, label_names):
    """
    Retrieve valid answer based on LLM response.
    """
    output = []
    invalid = 0
    for result in answer:

        if result == "":
            res = [("", 0)]
            output.append(res)
            continue
        res = []
        for line in result:
            this_line = []
            try:
                start = line.find('[')
                end = line.find(']', start) + 1
                list_str = line[start:end]
                this_dict = ast.literal_eval(list_str)
                for dic in this_dict:
                    answer = dic['answer']
                    if answer not in label_names:
                        answer = get_closest_label(answer, label_names)
                    confidence = dic['confidence']
                    this_line.append((answer, confidence))
                res.append(this_line)
            except:
                parts = line.split("},")
                for p in parts:
                    try:
                        ans = get_closest_label(p, label_names)
                        confidence = max(int(''.join(filter(str.isdigit, p))), 100)
                    except Exception:
                        confidence = 0
                    this_line.append((ans, confidence))
                    invalid += 1
                res.append(this_line)
        output.append(res)
    ("invalid number: {}".format(invalid))
    return output


def validate_syntax(old, label_names, format='[]'):
    """
    Check correctness of responses.
    """
    clean_t = old
    if format == '[]':
        start = clean_t.find('[')
        end = clean_t.find(']', start) + 1
        list_str = clean_t[start:end]
    else:
        start = clean_t.find('{')
        end = clean_t.find('}', start) + 1
        list_str = clean_t[start:end]
    try:
        result = ast.literal_eval(list_str)
    except Exception:
        return False, 'grammar error'
    try:
        first_answer = result[0]
        if not isinstance(first_answer, dict):
            return False, 'grammar error'
        else:
            answer = first_answer['answer']
            if answer in label_names:
                return True, 'success'
            else:
                return False, 'format error'
    except Exception:
        return False, 'format error'


def generate_correction_prompt(previous_prompt, label_names, correct_type='grammar error'):
    """
    Generate prompt for correcting grammar or format error.
    """
    prompt = "previous prompt: {} \n".format(previous_prompt)
    prompt += "Your previous output doesn't follow the format, please correct it\n"
    if correct_type == 'grammar error':
        prompt += "Your output should be a valid python object as a list of dictionaries"
    else:
        prompt += "Your previous answer is not a valid class.\n"
        prompt += "Your should only output categories from the following list: \n"
        prompt += "[" + ", ".join(label_names) + "]" + "\n"
    prompt += "New output here: "
    prompt = escape_curly_brackets(prompt)
    return prompt


def get_closest_label(input_string, label_names):
    """
    Find the label closest to the unrecognized answer in edit distance.
    """

    min_distance = float('inf')
    closest_label = None

    for label in label_names:
        distance = edit_distance(input_string, label)
        if distance < min_distance:
            min_distance = distance
            closest_label = label

    return closest_label


def topk_prompt(name, text, label_names, k=3):
    """
    Generate prompt template for top-k query.
    """
    prompt = text + "\n"
    prompt += "Task: \n"
    prompt += "There are following categories: \n"
    prompt += "[" + ", ".join(label_names) + "]" + "\n"
    if name == "tml1m":
        prompt += ("They denote the lower bound of age ranges. For example, 1 denotes 1-17 years old and 18 denotes 18-24 years old and so on. Note that young people watch old movies too and are more willing to rate movies. You should only reply with those given numbers."
)
    if name == "tacm12k":
        prompt += "The descriptions to the conferences are as follows:"
        prompt += "CIKM (Conference on Information and Knowledge Management): Focuses on research at the intersection of information retrieval, data management, and knowledge discovery. \nCOLT (Conference on Learning Theory): Dedicated to theoretical aspects of machine learning and statistical learning theory. \nICML (International Conference on Machine Learning): One of the top conferences for presenting cutting-edge research in machine learning. \nKDD (Knowledge Discovery and Data Mining): Premier conference for research on data mining, data science, and big data analytics. \nMobiCom (International Conference on Mobile Computing and Networking): Covers research on mobile systems, wireless networks, and mobile computing technologies. \nSIGCOMM (ACM Conference on Applications, Technologies, Architectures, and Protocols for Computer Communication): Leading venue for research in computer networking and communication systems. \nSIGIR (Special Interest Group on Information Retrieval): Premier forum for presenting research on information retrieval and search technologies. \nSIGMOD (Special Interest Group on Management of Data): Focuses on database systems and data management technologies. \nSODA (Symposium on Discrete Algorithms): Covers theoretical and practical aspects of algorithms and discrete mathematics. \nSOSP (Symposium on Operating Systems Principles): Top conference for innovations in operating systems and distributed systems. \nSPAA (Symposium on Parallelism in Algorithms and Architectures): Focuses on parallel computing in both theoretical and practical aspects. \nSTOC (Symposium on Theory of Computing): A flagship conference for theoretical computer science research. \nVLDB (Very Large Data Bases Conference): Leading venue for data management and large-scale data systems. \nWWW (The Web Conference): Covers web-related research including web mining, information retrieval, and web applications.\n"
    prompt = ("Question: {}. Give {} likely categories as a comma-separated "
              "list ordered from most to least likely together with a confidence "
              "ranging from 0 to 100, in the form of a list of dicts like "
              "[{{\"answer:\":<answer_here>, \"confidence\": <confidence_here>}}]."
              "Do not output anything else.        \
").format(prompt, k)
    prompt = escape_curly_brackets(prompt)
    return prompt


def escape_curly_brackets(text):
    """
    Replace single curly brackets with double curly brackets.
    """
    text = re.sub(r'(?<!{){(?!{)', '{{', text)
    text = re.sub(r'(?<!})}(?!})', '}}', text)
    return text


def edit_distance(s1, s2):
    if len(s1) > len(s2):
        s1, s2 = s2, s1

    distances = range(len(s1) + 1)
    for i2, c2 in enumerate(s2):
        distances_ = [i2+1]
        for i1, c1 in enumerate(s1):
            if c1 == c2:
                distances_.append(distances[i1])
            else:
                distances_.append(1 + min((distances[i1], distances[i1 + 1], distances_[-1])))
        distances = distances_
    return distances[-1]
