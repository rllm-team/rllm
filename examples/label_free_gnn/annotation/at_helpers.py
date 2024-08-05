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


def generate_prompts(texts, select_idx, label_names, k=3):
    """
    Generate prompt list based on raw text.
    """
    prompts = ["" for _ in range(len(texts))]
    for i, text in enumerate(texts):
        if i in select_idx:
            prompts[i] = topk_prompt(text, label_names, k)

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
            output = llm.predict(prompt)
            print('response:', output)
            ok, error_type = validate_syntax(output, label_names)
            if not ok:
                print('correcting', i, j)
                correction_prompt = generate_correction_prompt(prompt, label_names, error_type)
                output = llm.predict(correction_prompt)
                print('response:', output)
            res.append(output)
        responses.append(res)

    return responses


def collect_answers(answer, label_names):
    """
    Retrieve valid answer based on LLM  response.
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
                # if no error, retrieve all dicts in a list
                start = line.find('[')
                end = line.find(']', start) + 1  # +1 to include the closing bracket
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
                # if error, split the result based on },
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


def topk_prompt(text, label_names, k=3):
    """
    Generate prompt template for top-k query.
    """
    prompt = text + "\n"
    prompt += "Task: \n"
    prompt += "There are following categories: \n"
    prompt += "[" + ", ".join(label_names) + "]" + "\n"
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
